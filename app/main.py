"""Edge QC classifier inference service with two-process hot model swapping.

Public surface (see ``/docs``):

* ``POST /infer``            -- classify a feature vector with a client timeout
* ``POST /admin/models``     -- upload a classifier artifact and hot-swap it in
* ``GET  /admin/versions``   -- version registry
* ``GET  /admin/status``     -- active version, in-flight counts, process states
* ``GET  /healthz``          -- liveness

The invariants live in ``app/process_manager.py``.  In short: every model
version runs in its own OS process, at most two such processes exist at once,
requests are pinned to the version active at admission, and a retired version
only releases its slot once every inference it actually accepted has finished
inside its worker -- regardless of whether the client is still waiting.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.models import ModelValidationError, load_model, save_artifact, validate_artifact
from app.process_manager import (
    MAINTENANCE_DRAINING,
    MAINTENANCE_FAILED,
    MAINTENANCE_STOPPED,
    STATE_DRAINING,
    STATE_IDLE,
    VERSION_ACTIVE,
    VERSION_FAILED,
    VERSION_LOADING,
    VERSION_RETIRED,
    VERSION_STOPPED,
    FeatureMismatch,
    MaintenanceConflict,
    NoActiveModel,
    NoCapacity,
    Registry,
    ServiceStopping,
    SwapConflict,
    WorkerHandle,
)

logger = logging.getLogger("qc.service")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
MODEL_DIR = DATA_DIR / "models"
STAGING_DIR = DATA_DIR / "staging"
WORK_DIR = DATA_DIR / "workers"
DEFAULT_TIMEOUT_MS = int(os.environ.get("DEFAULT_INFER_TIMEOUT_MS", "5000"))
MAX_TIMEOUT_MS = 600_000
# How long an upload may wait for a draining slot to be released before 429.
SWAP_WAIT_SECONDS = float(os.environ.get("SWAP_WAIT_SECONDS", "3"))
# How long shutdown waits for accepted inference to finish before killing workers.
SHUTDOWN_GRACE_SECONDS = float(os.environ.get("SHUTDOWN_GRACE_SECONDS", "20"))
# How long a maintenance stop waits for accepted inference and in-progress swaps
# to settle before declaring the stop failed.  The stop never kills unsettled
# work: on expiry it reports failure and leaves the processes alone.
MAINTENANCE_SETTLE_TIMEOUT_SECONDS = float(
    os.environ.get("MAINTENANCE_SETTLE_TIMEOUT_SECONDS", "300")
)
# How long each worker process gets to exit during a maintenance stop.
MAINTENANCE_STOP_TIMEOUT_SECONDS = float(
    os.environ.get("MAINTENANCE_STOP_TIMEOUT_SECONDS", "10")
)
MAINTENANCE_POLL_SECONDS = 0.05
# How long POST /admin/maintenance/stop waits by default before returning 504.
DEFAULT_STOP_WAIT_SECONDS = float(os.environ.get("DEFAULT_STOP_WAIT_SECONDS", "20"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class InferRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector for the classifier")
    timeout_ms: Optional[int] = Field(
        None,
        ge=1,
        le=MAX_TIMEOUT_MS,
        description=(
            "How long the client is willing to wait, in milliseconds. On expiry the "
            "API returns 504; the model process keeps computing to completion."
        ),
    )
    timeout_seconds: Optional[float] = Field(
        None,
        gt=0,
        le=MAX_TIMEOUT_MS / 1000.0,
        description=(
            "Same as timeout_ms, expressed in seconds (e.g. 0.2). If both are "
            "provided, timeout_ms wins."
        ),
    )
    slow_ms: int = Field(
        0,
        ge=0,
        le=600_000,
        description=(
            "Demo hook: ask the model process to spend this long on the computation "
            "before scoring. Use with a small timeout_ms to demonstrate the 504 / "
            "drain behaviour."
        ),
    )

    @field_validator("features")
    @classmethod
    def _features_present(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("features must contain at least one value")
        return value


class InferResponse(BaseModel):
    version: str
    label: str
    score: float
    inference_id: str
    elapsed_ms: float


class TimeoutResponse(BaseModel):
    detail: Dict[str, Any]


class ModelUpload(BaseModel):
    version: str = Field(..., description="Unique version id, e.g. 'v3'")
    weights: List[List[float]] = Field(..., description="(n_classes x n_features) matrix")
    bias: List[float] = Field(..., description="Bias vector of length n_classes")
    labels: List[str] = Field(..., description="Class label for each weight row")
    test_features: Optional[List[float]] = Field(
        None,
        description="Vector used for the pre-activation smoke test; defaults to all zeros",
    )
    expected_active_version: Optional[str] = Field(
        None,
        description=(
            "Optimistic-concurrency guard: swap only if this is still the active "
            "version. Of concurrent swaps away from the same old version at most "
            "one succeeds."
        ),
    )
    slow_load_ms: int = Field(
        0,
        ge=0,
        le=60_000,
        description="Demo hook: delay worker readiness to observe the old version serving.",
    )
    simulate_load_failure: bool = Field(
        False,
        description="Demo hook: make the worker report a load failure after starting.",
    )


class SlotView(BaseModel):
    slot_id: str
    version: Optional[str]
    process_state: str
    pid: Optional[int]
    in_flight: int
    accepted: int
    completed: int
    alive: Optional[bool]


class MaintenanceView(BaseModel):
    phase: str
    generation: int
    stop_id: Optional[str] = None
    reason: Optional[str]
    error: Optional[str]
    started_at: Optional[float]
    stopped_at: Optional[float]
    failed_at: Optional[float]
    changed_at: Optional[float]
    processes_verified: bool
    admission_open: bool
    unsettled_inference: int
    unsettled_swaps: int
    unsettled_total: int
    slots_in_use: int
    safe_to_service: bool


class MaintenanceRequest(BaseModel):
    reason: Optional[str] = Field(
        None,
        max_length=500,
        description="Free-form note recorded with the stop (e.g. work order id)",
    )
    wait_seconds: float = Field(
        DEFAULT_STOP_WAIT_SECONDS,
        ge=0,
        le=300,
        description=(
            "How long this HTTP call waits for the stop to settle. On expiry the "
            "call returns 504 but the stop keeps running in the background; it is "
            "never auto-resumed."
        ),
    )


class MaintenanceStartResponse(BaseModel):
    phase: str
    generation: int
    admission_open: bool
    draining: bool
    safe_to_service: bool
    maintenance: MaintenanceView


class MaintenanceWaitResponse(BaseModel):
    """Outcome of waiting for a stop attempt, valid for one generation only."""

    generation: int
    phase: str
    safe_to_service: bool
    stale: bool
    wait_seconds: float
    maintenance: MaintenanceView
    detail: Dict[str, Any] = Field(default_factory=dict)


class MaintenanceResumeResponse(BaseModel):
    phase: str
    generation: int
    admission_open: bool
    active_version: Optional[str]
    note: str
    maintenance: MaintenanceView


class StatusView(BaseModel):
    active_version: Optional[str]
    version: Optional[str]
    total_in_flight: int
    total_accepted: int
    total_completed: int
    slots_in_use: int
    capacity: int
    slots: List[SlotView]
    versions: List[Dict[str, Any]]
    maintenance: MaintenanceView


class SwapResult(BaseModel):
    version: str
    previous_version: Optional[str]
    slot_id: str
    state: str
    process_state: str
    pid: Optional[int]
    activated_at: float
    test_label: str
    load_seconds: float


# ---------------------------------------------------------------------------
# Inference service
# ---------------------------------------------------------------------------


class InferenceService:
    def __init__(self) -> None:
        self.manager: Any = None
        self.registry: Registry
        self.handles: Dict[str, WorkerHandle] = {}
        self.client: Optional[httpx.AsyncClient] = None
        self._drain_task: Optional[asyncio.Task] = None
        self._closing = False
        self._draining: set[str] = set()
        self._background: set[asyncio.Task] = set()
        self._swap_lock = asyncio.Lock()  # serialises swaps inside this process
        self._lock_file: Any = None
        # Maintenance stop coordination.  ``_stop_task`` outlives the HTTP
        # request that started it: a client disconnect or timeout must not
        # abort a stop halfway, nor do anything that could be mistaken for
        # success.  ``_active_swaps`` counts swap claims held by this process so
        # the stop can wait for a version replacement already under way.
        self._stop_task: Optional[asyncio.Task] = None
        self._stop_lock = asyncio.Lock()
        # Serialises destructive worker-process operations (drain loop vs the
        # maintenance coordinator) so a process is never stopped twice.
        self._proc_lock = asyncio.Lock()

    # -- lifecycle -------------------------------------------------------
    async def startup(self) -> None:
        for path in (DATA_DIR, MODEL_DIR, STAGING_DIR, WORK_DIR):
            path.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(DATA_DIR / "api.lock", "a+")
        try:
            import fcntl

            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # pragma: no cover - non-POSIX
            pass
        except OSError as exc:
            raise RuntimeError(
                "another API process already owns this data directory; the "
                "service must run as a single process so the two-slot budget "
                "has a single owner"
            ) from exc
        self.manager = _new_manager()
        self.registry = Registry(self.manager)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=None, write=None, pool=None),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
        )
        self._drain_task = asyncio.create_task(self._drain_loop())
        bootstrap_version = os.environ.get("BOOTSTRAP_VERSION")
        bootstrap_model = os.environ.get("BOOTSTRAP_MODEL")
        if bootstrap_version and bootstrap_model and Path(bootstrap_model).is_file():
            try:
                await self.swap_in_path(
                    version=bootstrap_version,
                    source_path=bootstrap_model,
                    expected_active=None,
                    test_features=None,
                    slow_load_ms=0,
                    simulate_load_failure=False,
                )
                logger.info("bootstrapped model %s", bootstrap_version)
            except Exception:
                logger.exception(
                    "bootstrap model %s failed; service starts without a model",
                    bootstrap_version,
                )

    async def shutdown(self) -> None:
        self._closing = True
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._drain_task
            self._drain_task = None
        # A stop that is still running is abandoned, not "completed": process
        # shutdown is not a maintenance certification.
        if self._stop_task is not None:
            self._stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._stop_task
            self._stop_task = None
        # Let accepted inference settle so no in-flight work is killed halfway
        # and no worker is left orphaned.  Bounded so shutdown cannot hang.
        if self._background:
            logger.info("waiting for %d in-flight inference call(s)", len(self._background))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*self._background, return_exceptions=True),
                    timeout=SHUTDOWN_GRACE_SECONDS,
                )
        for task in list(self._background):
            task.cancel()
        for handle in list(self.handles.values()):
            await asyncio.to_thread(handle.stop, 5.0)
        self.handles.clear()
        if self.client is not None:
            await self.client.aclose()
            self.client = None
        if self.manager is not None:
            with contextlib.suppress(Exception):
                self.manager.shutdown()
            self.manager = None
        if self._lock_file is not None:
            with contextlib.suppress(Exception):
                self._lock_file.close()
            self._lock_file = None

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _artifact_path(version: str) -> Path:
        return MODEL_DIR / f"{version}.json"

    def _persist_artifact(self, version: str, source_path: str) -> None:
        """Copy the promoted artifact into the durable per-version location."""
        target = self._artifact_path(version)
        tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
        with open(source_path, "rb") as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        os.replace(tmp, target)

    async def _reserve_slot(self, version: str) -> str:
        """Reserve a process slot for a candidate, waiting briefly for a drain.

        A slot that is draining with zero in-flight work is about to be freed
        by the drain loop; waiting a couple of seconds avoids spurious 429s
        without ever exceeding two processes.  If the drain is held up by real
        in-flight work the wait expires and the caller gets 429.
        """
        deadline = time.monotonic() + SWAP_WAIT_SECONDS
        while True:
            try:
                return self.registry.reserve_any_slot(version)
            except SwapConflict as exc:
                # e.g. the version id was registered concurrently, or a retry
                # of a failed id raced a live one.
                raise HTTPException(
                    status_code=409,
                    detail={"error": "version_exists", "message": str(exc)},
                ) from exc
            except NoCapacity as exc:
                if time.monotonic() >= deadline:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "capacity_exhausted",
                            "message": str(exc),
                            "active_version": self.registry.active_version(),
                        },
                        headers={"Retry-After": "1"},
                    ) from exc
                await asyncio.sleep(0.05)

    async def _salvage_candidate(
        self, version: str, handle: WorkerHandle, reason: str
    ) -> None:
        """Stop a failed candidate, wait for exit, then free its slot."""
        self.handles.pop(version, None)
        await asyncio.to_thread(handle.stop, 5.0)
        self.registry.fail_candidate(version, reason)
        logger.warning("candidate %s failed: %s", version, reason)

    async def _test_candidate(
        self, version: str, test_features: Optional[List[float]]
    ) -> Dict[str, Any]:
        handle = self.handles[version]
        # Test against the exact artifact file this candidate process loaded,
        # not a shared per-version path (concurrent uploads use staged files).
        artifact = load_model(handle.model_path, version=version)
        features = (
            list(test_features)
            if test_features is not None
            else [0.0] * artifact.n_features
        )
        if len(features) != artifact.n_features:
            raise RuntimeError(
                f"test_features must have {artifact.n_features} values, got {len(features)}"
            )
        assert self.client is not None
        resp = await self.client.post(
            handle.inference_url(),
            json={"features": features},
            timeout=httpx.Timeout(10.0),
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"candidate smoke test failed with HTTP {resp.status_code}: {resp.text[:300]}"
            )
        body = resp.json()
        if body.get("version") != version:
            raise RuntimeError("candidate reported a mismatched version")
        if body.get("label") not in artifact.labels:
            raise RuntimeError("candidate returned a label outside the label set")
        return body

    # -- swap ------------------------------------------------------------
    async def swap_in_path(
        self,
        version: str,
        source_path: str,
        expected_active: Optional[str],
        test_features: Optional[List[float]],
        slow_load_ms: int,
        simulate_load_failure: bool,
    ) -> SwapResult:
        """Load a candidate in a fresh process, smoke-test it, then promote it.

        The old version keeps serving until the promotion is a single atomic
        registry operation.  On any failure the candidate is stopped, its slot
        freed, and the old version is left untouched.
        """
        # Snapshot the active version *before* waiting on the in-process swap
        # lock.  ``expected_active`` replaces the snapshot when the caller
        # supplies one.  Competing swaps therefore carry the same old version,
        # and the registry claim (keyed by that version) lets exactly one win.
        observed = (
            expected_active
            if expected_active is not None
            else self.registry.active_version()
        )
        # Fail fast on a known-live version id before taking the swap claim, so
        # the common duplicate-upload case never disturbs the claim table.
        existing = self.registry.version_info(version)
        if existing is not None and existing["state"] not in (
            VERSION_FAILED,
            VERSION_STOPPED,
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "version_exists",
                    "message": f"version {version!r} already exists ({existing['state']})",
                },
            )
        token = uuid.uuid4().hex
        # Claim outside the in-process lock so that a second swap observing the
        # same old version fails immediately instead of queueing behind the
        # first (which may be slow to load) and then failing anyway.
        try:
            old_version = self.registry.begin_swap(observed, token)
        except ServiceStopping as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_stopping",
                    "phase": exc.phase,
                    "message": str(exc),
                },
            ) from exc
        except SwapConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "swap_conflict", "message": str(exc)},
            ) from exc

        async with self._swap_lock:
            slot_id: Optional[str] = None
            handle: Optional[WorkerHandle] = None
            committed = False
            try:
                # Re-check under the lock: a concurrent request may have
                # registered this version while we waited.
                slot_id = await self._reserve_slot(version)

                handle = WorkerHandle(
                    slot_id=slot_id,
                    version=version,
                    model_path=source_path,
                    work_dir=str(WORK_DIR),
                )
                self.handles[version] = handle
                handle.spawn(
                    slow_load_ms=slow_load_ms,
                    simulate_load_failure=simulate_load_failure,
                )
                self.registry.set_pid(version, handle.pid or -1)

                load_timeout = 30.0 + max(0, slow_load_ms) / 1000.0
                try:
                    ready = await asyncio.to_thread(handle.wait_ready, load_timeout)
                except Exception as exc:
                    await self._salvage_candidate(version, handle, str(exc))
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "candidate_load_failed",
                            "message": str(exc),
                            "fallback_version": self.registry.active_version(),
                        },
                    ) from exc
                if not ready.get("ok"):
                    reason = str(ready.get("error") or "worker reported load failure")
                    await self._salvage_candidate(version, handle, reason)
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "candidate_load_failed",
                            "message": reason,
                            "fallback_version": self.registry.active_version(),
                        },
                    )
                if ready.get("n_features") is not None:
                    self.registry.set_model_shape(
                        version,
                        int(ready["n_features"]),
                        int(ready.get("n_classes", 0)),
                    )

                try:
                    test_body = await self._test_candidate(version, test_features)
                except Exception as exc:
                    await self._salvage_candidate(
                        version, handle, f"smoke test failed: {exc}"
                    )
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "candidate_test_failed",
                            "message": str(exc),
                            "fallback_version": self.registry.active_version(),
                        },
                    ) from exc

                self.registry.mark_candidate_tested(version)
                try:
                    self.registry.commit_swap(version, old_version, token)
                except SwapConflict as exc:
                    await self._salvage_candidate(version, handle, str(exc))
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "error": "swap_conflict",
                            "message": str(exc),
                            "active_version": self.registry.active_version(),
                        },
                    ) from exc
                committed = True
                # Keep a durable copy of what is now serving.  Staged uploads
                # are cleaned up by the endpoint; this snapshot is what a
                # future restart can reload.
                try:
                    self._persist_artifact(version, source_path)
                except Exception:
                    logger.exception("failed to persist artifact for %s", version)

                new_info = self.registry.version_info(version)
                assert new_info is not None
                logger.info(
                    "activated model %s in %s (previous=%s, pid=%s)",
                    version,
                    slot_id,
                    old_version,
                    handle.pid,
                )
                return SwapResult(
                    version=version,
                    previous_version=old_version,
                    slot_id=slot_id,
                    state=new_info["state"],
                    process_state=new_info["process_state"],
                    pid=handle.pid,
                    activated_at=float(new_info["activated_at"]),
                    test_label=test_body["label"],
                    load_seconds=float(ready.get("load_seconds", 0.0)),
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("swap of %s failed", version)
                raise HTTPException(
                    status_code=500,
                    detail={"error": "swap_failed", "message": str(exc)},
                ) from exc
            finally:
                if not committed:
                    if handle is not None and version in self.handles:
                        await self._salvage_candidate(version, handle, "swap aborted")
                    elif slot_id is not None:
                        self.registry.fail_candidate(version, "swap aborted")
                self.registry.end_swap(old_version, token)

    # -- maintenance shutdown --------------------------------------------
    async def request_stop(self, reason: Optional[str]) -> tuple[str, int]:
        """Close admission and start (or join) a maintenance stop.

        The coordinator task is deliberately detached from the HTTP request:
        if the caller times out or disconnects while waiting, the stop keeps
        running to a real conclusion and is never auto-resumed.  A second
        request while the stop is draining joins the existing attempt rather
        than bumping the generation (which would orphan the first waiter).
        """
        async with self._stop_lock:
            current = self.registry.maintenance_view()
            phase = current["phase"]
            if phase in (MAINTENANCE_DRAINING, MAINTENANCE_STOPPED):
                token = current["stop_id"]
                generation = current["generation"]
                if phase == MAINTENANCE_STOPPED:
                    return token, generation
                # Joining a draining attempt: make sure its coordinator lives.
                if self._stop_task is None or self._stop_task.done():
                    self._stop_task = asyncio.create_task(
                        self._run_maintenance(token, generation, reason)
                    )
                return token, generation
            token, generation = self.registry.begin_maintenance(reason)
            if self._stop_task is not None and not self._stop_task.done():
                # Defensive: begin_maintenance refuses while draining, so a
                # live task should not coexist.  Cancel rather than orphan.
                self._stop_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._stop_task
            self._stop_task = asyncio.create_task(
                self._run_maintenance(token, generation, reason)
            )
            return token, generation

    async def wait_for_stop(
        self, token: str, generation: int, wait_seconds: float
    ) -> Dict[str, Any]:
        """Wait for this stop generation to settle, bounded by ``wait_seconds``.

        Returns the generation-scoped outcome.  A waiter that is still draining
        when the wait expires receives ``stale``/``safe_to_service`` false; it
        can poll again with the same ``stop_id`` and generation.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, wait_seconds)
        while True:
            view = self.registry.maintenance_outcome(token, generation)
            if view["stale"] or view["phase"] in (
                MAINTENANCE_STOPPED,
                MAINTENANCE_FAILED,
            ):
                return view
            if loop.time() >= deadline:
                return view
            await asyncio.sleep(MAINTENANCE_POLL_SECONDS)

    async def _run_maintenance(
        self, token: str, generation: int, reason: Optional[str]
    ) -> None:
        logger.info("maintenance stop generation=%d started (reason=%r)", generation, reason)
        loop = asyncio.get_running_loop()
        try:
            settled = await self._await_settled(token, generation, loop)
            if not settled:
                return  # failed / stale; registry already records why
            stopped = await self._stop_all_processes(token, generation)
            if not stopped:
                return
            self.registry.mark_processes_verified(token, generation)
            if self.registry.finish_maintenance(token, generation):
                view = self.registry.maintenance_view()
                logger.info(
                    "maintenance stop generation=%d complete: slots_in_use=%d safe=%s",
                    generation,
                    view["slots_in_use"],
                    view["safe_to_service"],
                )
            else:
                view = self.registry.maintenance_view()
                logger.error(
                    "maintenance stop generation=%d could not certify idle "
                    "(slots_in_use=%d unsettled=%d)",
                    generation,
                    view["slots_in_use"],
                    view["unsettled_total"],
                )
                self.registry.fail_maintenance(
                    token,
                    generation,
                    "processes exited but the machine is not provably idle "
                    f"(slots_in_use={view['slots_in_use']}, "
                    f"unsettled={view['unsettled_total']})",
                )
        except asyncio.CancelledError:
            logger.warning("maintenance stop generation=%d was cancelled", generation)
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("maintenance stop generation=%d crashed", generation)
            self.registry.fail_maintenance(
                token, generation, f"stop coordinator error: {exc}"
            )

    async def _await_settled(
        self, token: str, generation: int, loop: asyncio.AbstractEventLoop
    ) -> bool:
        """Wait for accepted inference and in-flight swaps to actually settle.

        Nothing is killed while work is outstanding: a 504'd or disconnected
        request keeps its in-flight counter until the worker call returns, and
        an already-started swap keeps its registry claim until it commits or
        aborts.  On timeout the stop is marked failed and processes are left
        untouched.
        """
        deadline = loop.time() + max(0.0, MAINTENANCE_SETTLE_TIMEOUT_SECONDS)
        while True:
            view = self.registry.maintenance_outcome(token, generation)
            if view["stale"] or view["phase"] != MAINTENANCE_DRAINING:
                return False
            if view["unsettled_inference"] == 0 and view["unsettled_swaps"] == 0:
                return True
            if loop.time() >= deadline:
                self.registry.fail_maintenance(
                    token,
                    generation,
                    "timed out waiting for outstanding work to settle "
                    f"(inference={view['unsettled_inference']}, "
                    f"swaps={view['unsettled_swaps']})",
                )
                logger.error(
                    "maintenance stop generation=%d timed out with unsettled work; "
                    "processes were left running and the machine is NOT serviceable",
                    generation,
                )
                return False
            await asyncio.sleep(MAINTENANCE_POLL_SECONDS)

    async def _stop_all_processes(self, token: str, generation: int) -> bool:
        """Stop every worker process, verify exit, then release every slot.

        Runs under ``_proc_lock`` so the drain loop cannot concurrently stop or
        release the same process.  Any process that cannot be proven exited
        turns the stop into a failure -- being alive is never reported safe.
        """
        async with self._proc_lock:
            if self.registry.maintenance_view()["generation"] != generation:
                return False
            for version, handle in list(self.handles.items()):
                try:
                    await asyncio.to_thread(
                        handle.stop, MAINTENANCE_STOP_TIMEOUT_SECONDS
                    )
                    await asyncio.to_thread(
                        handle.wait_exit, MAINTENANCE_STOP_TIMEOUT_SECONDS
                    )
                except Exception:
                    logger.exception("failed to stop worker for %s", version)
                if not handle.exited():
                    self.registry.fail_maintenance(
                        token,
                        generation,
                        f"worker process for {version} is still alive after stop",
                    )
                    logger.error(
                        "maintenance stop generation=%d failed: %s still alive",
                        generation,
                        version,
                    )
                    return False
                info = self.registry.version_info(version) or {}
                slot_id = info.get("slot_id")
                if slot_id:
                    self.registry.release_after_stop(
                        slot_id, version, process_exited=True
                    )
                self.handles.pop(version, None)

            # Slots whose handle is already gone (a crashed worker the drain
            # loop reaped) still need their bookkeeping freed -- but only when
            # the recorded pid is provably gone.
            for slot in self.registry.snapshot()["slots"]:
                version = slot["version"]
                if version is None:
                    continue
                if self._pid_alive(slot["pid"]):
                    self.registry.fail_maintenance(
                        token,
                        generation,
                        f"slot {slot['slot_id']} still has live pid {slot['pid']}",
                    )
                    return False
                self.registry.release_after_stop(
                    slot["slot_id"], version, process_exited=True
                )
                self.handles.pop(version, None)

            view = self.registry.maintenance_view()
            if view["slots_in_use"] != 0:
                self.registry.fail_maintenance(
                    token,
                    generation,
                    f"{view['slots_in_use']} slot(s) still occupied after stopping",
                )
                return False
            return True

    @staticmethod
    def _pid_alive(pid: Optional[int]) -> bool:
        if not pid or pid < 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def start_resume(self) -> Dict[str, Any]:
        """Reopen admission.  Workers stay down and inference stays 503."""
        generation = self.registry.resume_maintenance()
        logger.info("maintenance resumed at generation=%d", generation)
        return self.registry.maintenance_view()

    # -- draining --------------------------------------------------------
    async def _drain_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(0.05)
                await self.drain_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("drain loop error")

    async def drain_once(self) -> None:
        async with self._proc_lock:
            await self._drain_once_locked()

    async def _drain_once_locked(self) -> None:
        snapshot = self.registry.snapshot()
        for version, info in snapshot["versions"].items():
            handle = self.handles.get(version)
            # Unexpected worker death: a crashed worker cannot be part of the
            # two-process budget any more, but its slot must still be reclaimed
            # so the machine does not appear full forever.
            if handle is not None and not handle.alive():
                self._reap_dead_worker(version, info)

            if info["state"] not in (VERSION_RETIRED, VERSION_FAILED):
                continue
            if info["process_state"] not in (STATE_DRAINING, STATE_IDLE):
                continue
            slot = next(
                (s for s in snapshot["slots"] if s["slot_id"] == info["slot_id"]),
                None,
            )
            if slot is None or slot["version"] != version:
                continue
            if version in self._draining:
                continue
            # The drain gate: only stop the process once no accepted request
            # is still running inside it.
            if self.registry.version_in_flight(version) > 0:
                continue
            self._draining.add(version)
            try:
                if handle is not None:
                    await asyncio.to_thread(handle.stop, 10.0)
                    await asyncio.to_thread(handle.wait_exit, 10.0)
                released = self.registry.release_slot(
                    slot["slot_id"], version, process_exited=True
                )
                if released:
                    self.handles.pop(version, None)
                    logger.info("drained %s; slot %s released", version, slot["slot_id"])
            except Exception:
                logger.exception("failed to drain %s", version)
            finally:
                self._draining.discard(version)

    def _reap_dead_worker(self, version: str, info: Dict[str, Any]) -> None:
        """Reclaim bookkeeping for a worker process that died unexpectedly."""
        self.handles.pop(version, None)
        if info["state"] in (VERSION_ACTIVE, VERSION_LOADING):
            self.registry.settle_dead_worker(version)
            if self.registry.fail_active(version, "worker process exited unexpectedly"):
                logger.error("active model %s died; service has no active model", version)
        elif info["state"] in (VERSION_RETIRED, VERSION_FAILED):
            self.registry.settle_dead_worker(version)
            with contextlib.suppress(Exception):
                self.registry.release_slot(info["slot_id"], version, process_exited=True)

    # -- inference -------------------------------------------------------
    async def infer(
        self, features: List[float], timeout_ms: int, slow_ms: int = 0
    ) -> InferResponse:
        try:
            version, _slot_id = self.registry.accept_work(len(features))
        except ServiceStopping as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_stopping",
                    "phase": exc.phase,
                    "message": str(exc),
                },
            ) from exc
        except FeatureMismatch as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "feature_dimension_mismatch",
                    "message": str(exc),
                    "version": exc.version,
                    "expected_features": exc.expected,
                    "received_features": len(features),
                },
            ) from exc
        except NoActiveModel as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": "no_active_model", "message": str(exc)},
            ) from exc

        handle = self.handles.get(version)
        if handle is None:  # pragma: no cover - registry/handle invariant
            self.registry.complete_work(version)
            raise HTTPException(
                status_code=503,
                detail={"error": "worker_missing", "message": f"worker for {version} is gone"},
            )

        payload: Dict[str, Any] = {"features": features}
        if slow_ms:
            payload["slow_ms"] = slow_ms
        inference_id = uuid.uuid4().hex

        # Shield the worker call from this request's timeout/cancellation.  The
        # in-flight counter must only clear when the model process is actually
        # done; a 504 or disconnected client must not fake completion.
        task = asyncio.create_task(self._proxy_inference(version, handle, payload))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - started) * 1000.0
            logger.warning(
                "inference on %s exceeded %dms (request %s); worker continues",
                version,
                timeout_ms,
                inference_id,
            )
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "inference_timeout",
                    "version": version,
                    "timeout_ms": timeout_ms,
                    "elapsed_ms": round(elapsed, 3),
                    "note": (
                        "The client stopped waiting, but the model process continues "
                        "the computation; its slot is not released until that work "
                        "actually finishes."
                    ),
                },
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "worker_unreachable",
                    "version": version,
                    "message": str(exc),
                },
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error": "worker_error", "version": version, "message": str(exc)},
            ) from exc

        elapsed = (time.perf_counter() - started) * 1000.0
        return InferResponse(
            version=version,
            label=result["label"],
            score=float(result["score"]),
            inference_id=inference_id,
            elapsed_ms=round(elapsed, 3),
        )

    async def _proxy_inference(
        self, version: str, handle: WorkerHandle, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            assert self.client is not None
            resp = await self.client.post(handle.inference_url(), json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"worker {version} returned HTTP {resp.status_code}: {resp.text[:300]}"
                )
            return resp.json()
        finally:
            # Runs once, when the worker call actually settles.  This is the
            # only place a request leaves the drain gate.
            self.registry.complete_work(version)

    # -- status ----------------------------------------------------------
    def status(self) -> StatusView:
        snapshot = self.registry.snapshot()
        slots: List[SlotView] = []
        total_in_flight = 0
        total_accepted = 0
        total_completed = 0
        active = snapshot["active_version"]
        for slot in snapshot["slots"]:
            handle = self.handles.get(slot["version"]) if slot["version"] else None
            alive = handle.alive() if handle is not None else None
            total_in_flight += int(slot["in_flight"])
            total_accepted += int(slot["accepted"])
            total_completed += int(slot["completed"])
            slots.append(
                SlotView(
                    slot_id=slot["slot_id"],
                    version=slot["version"],
                    process_state=slot["process_state"],
                    pid=slot["pid"],
                    in_flight=int(slot["in_flight"]),
                    accepted=int(slot["accepted"]),
                    completed=int(slot["completed"]),
                    alive=alive,
                )
            )
        return StatusView(
            active_version=active,
            version=active,
            total_in_flight=total_in_flight,
            total_accepted=total_accepted,
            total_completed=total_completed,
            slots_in_use=snapshot["slots_in_use"],
            capacity=snapshot["capacity"],
            slots=slots,
            versions=sorted(
                snapshot["versions"].values(), key=lambda v: v.get("created_at", 0)
            ),
            maintenance=MaintenanceView(**snapshot["maintenance"]),
        )

    def maintenance(self) -> MaintenanceView:
        return MaintenanceView(**self.registry.maintenance_view())


def _new_manager() -> Any:
    """Create the registry manager, preferring ``fork`` on POSIX.

    The manager server is started once during startup, before the API handles
    requests; using ``fork`` avoids re-importing the uvicorn entry point in a
    spawned child.  On platforms without ``fork`` the default context is used.
    """
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - Windows / spawn-only platforms
        ctx = multiprocessing.get_context()
    return ctx.Manager()


service = InferenceService()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.startup()
    try:
        yield
    finally:
        await service.shutdown()


app = FastAPI(
    title="Edge QC Classifier Inference Service",
    version="1.0.0",
    description=(
        "Linear classifier inference with live model replacement on an edge QC "
        "machine that can host at most **two model processes**.\n\n"
        "* `POST /admin/models` uploads `weights`, `bias` and `labels`; the new "
        "version is loaded in a free process, smoke-tested, then promoted. The "
        "old version keeps serving during load and keeps serving if the new one "
        "fails.\n"
        "* `POST /infer` returns the top-scoring label and the version. Requests "
        "are pinned to the version active when accepted. `timeout_ms` (or "
        "`timeout_seconds`) bounds the *client wait only*: on expiry you get 504 "
        "while the worker keeps working and keeps its slot.\n"
        "* `GET /admin/status` shows the active version, in-flight counts and each "
        "process slot's state (`loading` / `active` / `draining` / `idle`).\n\n"
        "**Demos**\n"
        "- Slow inference: `POST /infer` with `slow_ms: 3000`, `timeout_ms: 200`, "
        "then watch `/admin/status` and try uploading another version: while the "
        "drained version still works, only one slot is free.\n"
        "- Load failure: upload with `simulate_load_failure: true` (or malformed "
        "weights). You get 502 and the old version remains active.\n"
        "- Capacity: trigger a slow in-flight request, swap once, then try a third "
        "upload -> 429 until the slow work finishes and the slot is released.\n"
        "- Maintenance: `POST /admin/maintenance/stop` with `wait_seconds` refuses "
        "new inference/uploads, waits for already-accepted work (including 504'd "
        "requests) and any started swap to settle, stops both model processes, and "
        "returns `safe_to_service` only after every process is confirmed gone and "
        "both slots are idle. `POST /admin/maintenance/resume` reopens uploads; "
        "inference stays 503 until a new model is loaded."
    ),
    lifespan=lifespan,
)


@app.get("/healthz", tags=["ops"], summary="Liveness probe")
def healthz() -> Dict[str, Any]:
    # Deliberately always 200 while the process is alive: a maintenance stop
    # must not look like a failed container, or an orchestrator would restart
    # it and undo the stop.  The phase is exposed for observability only.
    active = service.registry.active_version()
    phase = service.registry.maintenance_phase()
    return {"ok": True, "active_version": active, "maintenance_phase": phase}


@app.post(
    "/infer",
    response_model=InferResponse,
    responses={
        504: {
            "model": TimeoutResponse,
            "description": "Client timeout; the worker keeps computing and holds its slot.",
        },
        503: {"description": "No active model"},
        502: {"description": "Worker unreachable or errored"},
    },
    tags=["inference"],
    summary="Classify a feature vector",
)
async def infer(req: InferRequest) -> InferResponse:
    if req.timeout_ms is not None:
        timeout_ms = req.timeout_ms
    elif req.timeout_seconds is not None:
        timeout_ms = max(1, int(round(req.timeout_seconds * 1000)))
    else:
        timeout_ms = DEFAULT_TIMEOUT_MS
    return await service.infer(req.features, timeout_ms, req.slow_ms)


@app.post(
    "/admin/models",
    response_model=SwapResult,
    status_code=201,
    responses={
        409: {"description": "Version exists or swap conflict / stale expectation"},
        429: {"description": "Both model process slots are still occupied"},
        502: {"description": "Candidate load or smoke test failed; old version kept"},
        422: {"description": "Artifact is structurally invalid"},
    },
    tags=["admin"],
    summary="Upload a classifier and hot-swap it in",
)
async def upload_model(payload: ModelUpload) -> SwapResult:
    try:
        service.registry.require_admission("model upload")
    except ServiceStopping as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_stopping",
                "phase": exc.phase,
                "message": str(exc),
            },
        ) from exc
    if not VERSION_RE.match(payload.version):
        raise HTTPException(
            status_code=422,
            detail="version must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        )
    try:
        validate_artifact(payload.model_dump())
    except ModelValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_artifact", "message": str(exc)},
        ) from exc

    target = STAGING_DIR / f"{payload.version}-{uuid.uuid4().hex[:12]}.json"
    try:
        save_artifact(str(target), payload.version, payload.model_dump())
    except (OSError, ModelValidationError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "artifact_write_failed", "message": str(exc)},
        ) from exc

    try:
        result = await service.swap_in_path(
            version=payload.version,
            source_path=str(target),
            expected_active=payload.expected_active_version,
            test_features=payload.test_features,
            slow_load_ms=payload.slow_load_ms,
            simulate_load_failure=payload.simulate_load_failure,
        )
    finally:
        # The worker process holds its own loaded copy in memory; the staging
        # file is only needed for the load.  Published artifacts live under
        # ``models/<version>.json`` and are never overwritten by an upload.
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
    return result


@app.get("/admin/versions", tags=["admin"], summary="Version registry")
def versions() -> Dict[str, Any]:
    snapshot = service.registry.snapshot()
    return {
        "active_version": snapshot["active_version"],
        "capacity": snapshot["capacity"],
        "slots_in_use": snapshot["slots_in_use"],
        "versions": sorted(
            snapshot["versions"].values(), key=lambda v: v.get("created_at", 0)
        ),
    }


@app.get(
    "/admin/status",
    response_model=StatusView,
    tags=["admin"],
    summary="Active version, in-flight counts, process states and stop phase",
)
def status() -> StatusView:
    return service.status()


@app.post(
    "/admin/maintenance/stop",
    response_model=MaintenanceWaitResponse,
    tags=["maintenance"],
    responses={
        409: {"description": "A stop is already draining or the service is stopped"},
        504: {
            "description": (
                "The stop was accepted but did not settle within wait_seconds. "
                "It keeps running; poll status with the returned stop_id. No "
                "process was killed and the machine is not reported safe."
            )
        },
    },
    summary="Refuse new work, drain accepted work, stop all model processes",
)
async def maintenance_stop(
    payload: Optional[MaintenanceRequest] = None,
) -> MaintenanceWaitResponse:
    reason = payload.reason if payload is not None else None
    wait_seconds = (
        payload.wait_seconds if payload is not None else DEFAULT_STOP_WAIT_SECONDS
    )
    try:
        token, generation = await service.request_stop(reason)
    except MaintenanceConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "maintenance_conflict", "message": str(exc)},
        ) from exc
    started = time.perf_counter()
    view = await service.wait_for_stop(token, generation, wait_seconds)
    elapsed = time.perf_counter() - started
    if view["stale"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "maintenance_stale",
                "message": (
                    "this stop generation was superseded; its outcome no longer "
                    "describes the machine"
                ),
                "stop_id": token,
                "generation": generation,
                "phase": view["phase"],
                "safe_to_service": False,
            },
        )
    if view["phase"] == MAINTENANCE_DRAINING:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "maintenance_still_draining",
                "stop_id": token,
                "generation": generation,
                "phase": view["phase"],
                "safe_to_service": False,
                "unsettled_inference": view["unsettled_inference"],
                "unsettled_swaps": view["unsettled_swaps"],
                "slots_in_use": view["slots_in_use"],
                "waited_seconds": round(elapsed, 3),
                "note": (
                    "The stop continues in the background and is not auto-resumed. "
                    "Poll GET /admin/status; never treat a timeout as safe."
                ),
            },
        )
    if view["phase"] == MAINTENANCE_FAILED or not view["safe_to_service"]:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "maintenance_failed",
                "stop_id": token,
                "generation": generation,
                "phase": view["phase"],
                "safe_to_service": False,
                "error_detail": view["error"],
                "slots_in_use": view["slots_in_use"],
                "unsettled_inference": view["unsettled_inference"],
                "unsettled_swaps": view["unsettled_swaps"],
                "note": "Do not service the machine; inspect status and logs.",
            },
        )
    return MaintenanceWaitResponse(
        generation=generation,
        phase=view["phase"],
        safe_to_service=True,
        stale=False,
        wait_seconds=round(elapsed, 3),
        maintenance=MaintenanceView(**view),
    )


@app.get(
    "/admin/maintenance/status",
    response_model=MaintenanceWaitResponse,
    tags=["maintenance"],
    summary="Stop phase, unsettled work, slot occupancy; safe only for this generation",
)
def maintenance_status(
    stop_id: Optional[str] = None, generation: Optional[int] = None
) -> MaintenanceWaitResponse:
    current = service.registry.maintenance_view()
    token = stop_id if stop_id is not None else current["stop_id"]
    gen = generation if generation is not None else current["generation"]
    view = service.registry.maintenance_outcome(token, gen)
    return MaintenanceWaitResponse(
        generation=gen,
        phase=view["phase"],
        safe_to_service=bool(view["safe_to_service"]),
        stale=bool(view["stale"]),
        wait_seconds=0.0,
        maintenance=MaintenanceView(**view),
    )


@app.post(
    "/admin/maintenance/resume",
    response_model=MaintenanceResumeResponse,
    tags=["maintenance"],
    responses={409: {"description": "Not stopped/failed, or still draining"}},
    summary="Leave maintenance: reopen uploads; inference stays 503 until a model loads",
)
def maintenance_resume() -> MaintenanceResumeResponse:
    try:
        view = service.start_resume()
    except MaintenanceConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "maintenance_conflict", "message": str(exc)},
        ) from exc
    return MaintenanceResumeResponse(
        phase=view["phase"],
        generation=view["generation"],
        admission_open=view["admission_open"],
        active_version=service.registry.active_version(),
        note=(
            "Model uploads are accepted again. Inference keeps returning 503 "
            "until a newly uploaded model has loaded and been promoted."
        ),
        maintenance=MaintenanceView(**view),
    )
