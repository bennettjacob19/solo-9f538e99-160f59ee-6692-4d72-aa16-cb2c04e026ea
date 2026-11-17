"""Cross-process model-version registry and worker process handles.

The machine may host **at most two model processes** at a time.  Every
published model version is served by its own OS process (a ``uvicorn`` server
started in a new process group).  This module owns the bookkeeping that
enforces the cap and the swap state machine.

Per version / slot::

    (vacant) --reserve--> loading --commit--> active --retire--> draining --exit--> idle
                               |
                               +--load/test failure--> failed (slot freed)

Rules encoded here:

* Two persistent slots (``slot-0``, ``slot-1``).  A slot is reusable only
  after its process has exited *and* every inference it accepted has actually
  finished.  Until then the slot is occupied, so a third load is refused.
* A slot holding a ``loading`` candidate and a slot holding the ``active``
  version are both occupied; only ``idle`` slots can be reserved.
* ``accept_work`` reads the active version and increments that slot's
  in-flight counter in one locked operation.  The returned version is the
  version the request must use, so an accepted request can never be moved to a
  different model by a swap that commits immediately afterwards.
* ``commit_swap`` atomically promotes the tested candidate and moves the old
  version to ``draining``.  The old process is stopped only by the drain
  coordinator after ``in_flight == 0``; only after the OS process exits is
  ``release_slot`` allowed.
* ``begin_swap`` claims a swap *keyed by the old active version*.  A second
  swap claiming the same old version while the first is still in progress is
  rejected before any process is started.  The API runs as a single process
  (enforced by a data-directory lock), so this one writer owns the budget.
* A maintenance stop flips ``maintenance_phase`` out of ``running`` in the
  same lock as ``accept_work``/``begin_swap``, so admission closes exactly
  once.  ``safe_to_service`` is only ever true for the current generation,
  after the coordinator has verified every worker process exited, both slots
  are idle, and no in-flight inference or open swap claim remains.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SLOT_COUNT = 2

STATE_LOADING = "loading"
STATE_ACTIVE = "active"
STATE_DRAINING = "draining"
STATE_IDLE = "idle"

VERSION_LOADING = "loading"
VERSION_ACTIVE = "active"
VERSION_RETIRED = "retired"
VERSION_FAILED = "failed"
VERSION_STOPPED = "stopped"

# Maintenance-shutdown phases.  ``running`` is the only phase that admits new
# inference and new model uploads.
MAINTENANCE_RUNNING = "running"
MAINTENANCE_DRAINING = "draining"
MAINTENANCE_STOPPED = "stopped"
MAINTENANCE_FAILED = "failed"

CLAIM_NONE = "__none__"


class NoCapacity(RuntimeError):
    """Both process slots are occupied (loading/active/draining)."""


class ServiceStopping(RuntimeError):
    """Maintenance shutdown has begun; new work must not be admitted."""

    def __init__(self, message: str, phase: str, generation: int) -> None:
        super().__init__(message)
        self.phase = phase
        self.generation = generation


class MaintenanceConflict(RuntimeError):
    """The requested maintenance transition is invalid for the current phase."""


class SwapConflict(RuntimeError):
    """The swap's old version is no longer current, or a swap is in progress."""


class NoActiveModel(RuntimeError):
    """Inference was requested before any model became active."""


class FeatureMismatch(ValueError):
    """The feature vector width does not match the active model."""

    def __init__(self, message: str, version: str, expected: int) -> None:
        super().__init__(message)
        self.version = version
        self.expected = expected


class Registry:
    """Manager-backed shared state; every operation is serialised by a lock."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._lock = manager.RLock()
        self._versions = manager.dict()
        self._slots = manager.dict()
        self._claims = manager.dict()
        now = time.time()
        self._state = manager.dict(
            {
                "active_version": None,
                # Maintenance shutdown state.  ``generation`` is bumped by every
                # stop attempt and every resume; a waiter may only report
                # "stopped" for the exact generation it started with, so a
                # stale waiter can never call the machine serviceable after a
                # later stop/resume interleaving.
                "maintenance_phase": MAINTENANCE_RUNNING,
                "maintenance_generation": 0,
                "maintenance_token": None,
                "maintenance_verified": False,
                "maintenance_reason": None,
                "maintenance_error": None,
                "maintenance_started_at": None,
                "maintenance_stopped_at": None,
                "maintenance_failed_at": None,
                "maintenance_changed_at": now,
            }
        )
        for i in range(SLOT_COUNT):
            slot_id = f"slot-{i}"
            self._slots[slot_id] = {
                "slot_id": slot_id,
                "version": None,
                "process_state": STATE_IDLE,
                "pid": None,
                "in_flight": 0,
                "accepted": 0,
                "completed": 0,
                "started_at": None,
                "idle_since": now,
                "generation": 0,
            }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            versions = {k: dict(v) for k, v in self._versions.items()}
            slots = {k: dict(v) for k, v in self._slots.items()}
            return {
                "active_version": self._state["active_version"],
                "capacity": SLOT_COUNT,
                "slots_in_use": sum(
                    1 for s in slots.values() if s["process_state"] != STATE_IDLE
                ),
                "slots": [slots[k] for k in sorted(slots)],
                "versions": versions,
                "claims": dict(self._claims),
                "maintenance": self._maintenance_view_locked(),
            }

    def _maintenance_view_locked(self) -> Dict[str, Any]:
        """Maintenance view computed under the registry lock.

        ``safe_to_service`` is deliberately strict: it requires the coordinator
        to have verified process exit (``processes_verified``) *for the current
        generation*, the stop phase to be fully reached, both slots idle, and
        zero unsettled work.  Callers may only report "safe" from this view.
        """
        unsettled_inference = sum(
            int(s["in_flight"]) for s in self._slots.values()
        )
        unsettled_swaps = len(self._claims)
        slots_in_use = sum(
            1 for s in self._slots.values() if s["process_state"] != STATE_IDLE
        )
        phase = self._state["maintenance_phase"]
        verified = bool(self._state["maintenance_verified"])
        return {
            "phase": phase,
            "generation": int(self._state["maintenance_generation"]),
            "stop_id": self._state["maintenance_token"],
            "reason": self._state["maintenance_reason"],
            "error": self._state["maintenance_error"],
            "started_at": self._state["maintenance_started_at"],
            "stopped_at": self._state["maintenance_stopped_at"],
            "failed_at": self._state["maintenance_failed_at"],
            "changed_at": self._state["maintenance_changed_at"],
            "processes_verified": verified,
            "admission_open": phase == MAINTENANCE_RUNNING,
            "unsettled_inference": unsettled_inference,
            "unsettled_swaps": unsettled_swaps,
            "unsettled_total": unsettled_inference + unsettled_swaps,
            "slots_in_use": slots_in_use,
            "safe_to_service": (
                phase == MAINTENANCE_STOPPED
                and verified
                and slots_in_use == 0
                and unsettled_inference == 0
                and unsettled_swaps == 0
            ),
        }

    def maintenance_view(self) -> Dict[str, Any]:
        with self._lock:
            return self._maintenance_view_locked()

    def maintenance_phase(self) -> str:
        with self._lock:
            return str(self._state["maintenance_phase"])

    def require_admission(self, action: str) -> None:
        """Reject new work once maintenance shutdown has been requested."""
        with self._lock:
            self._require_admission_locked(action)

    def _require_admission_locked(self, action: str) -> None:
        phase = self._state["maintenance_phase"]
        if phase != MAINTENANCE_RUNNING:
            generation = int(self._state["maintenance_generation"])
            raise ServiceStopping(
                f"maintenance shutdown is {phase}; {action} is not admitted",
                phase=phase,
                generation=generation,
            )

    def active_version(self) -> Optional[str]:
        with self._lock:
            return self._state["active_version"]

    def version_info(self, version: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            info = self._versions.get(version)
            return dict(info) if info is not None else None

    def version_in_flight(self, version: str) -> int:
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                return 0
            slot = self._slots.get(info["slot_id"])
            if slot is None or slot["version"] != version:
                return 0
            return int(slot["in_flight"])

    def has_capacity(self) -> bool:
        with self._lock:
            return any(
                s["process_state"] == STATE_IDLE for s in self._slots.values()
            )

    # ------------------------------------------------------------------
    # Maintenance shutdown
    # ------------------------------------------------------------------
    def begin_maintenance(self, reason: Optional[str]) -> Tuple[str, int]:
        """Close admission and enter the draining phase.

        Atomic with ``accept_work`` / ``begin_swap``: whichever side takes the
        registry lock first wins, so there is no window in which a request is
        admitted after the stop has been acknowledged.  A ``failed`` stop may
        be retried; a stop that is already ``draining`` or ``stopped`` may not
        be started again until it has been resumed.
        """
        with self._lock:
            phase = self._state["maintenance_phase"]
            if phase == MAINTENANCE_DRAINING:
                raise MaintenanceConflict("a maintenance shutdown is already draining")
            if phase == MAINTENANCE_STOPPED:
                raise MaintenanceConflict(
                    "the service is already stopped for maintenance; resume first"
                )
            generation = int(self._state["maintenance_generation"]) + 1
            token = uuid.uuid4().hex
            now = time.time()
            self._state["maintenance_generation"] = generation
            self._state["maintenance_phase"] = MAINTENANCE_DRAINING
            self._state["maintenance_token"] = token
            self._state["maintenance_verified"] = False
            self._state["maintenance_reason"] = reason
            self._state["maintenance_error"] = None
            self._state["maintenance_started_at"] = now
            self._state["maintenance_stopped_at"] = None
            self._state["maintenance_failed_at"] = None
            self._state["maintenance_changed_at"] = now
            return token, generation

    def mark_processes_verified(self, token: str, generation: int) -> bool:
        """Record that every worker process for this stop was proven exited."""
        with self._lock:
            if not self._maintenance_token_matches_locked(token, generation):
                return False
            if self._state["maintenance_phase"] != MAINTENANCE_DRAINING:
                return False
            self._state["maintenance_verified"] = True
            self._state["maintenance_changed_at"] = time.time()
            return True

    def finish_maintenance(self, token: str, generation: int) -> bool:
        """Promote draining -> stopped, but only when the machine is provably idle.

        Requires the coordinator's process-exit verification, zero slots in
        use, zero in-flight inference and zero open swap claims.  On a stale
        token/generation it changes nothing and returns False.
        """
        with self._lock:
            if not self._maintenance_token_matches_locked(token, generation):
                return False
            if self._state["maintenance_phase"] != MAINTENANCE_DRAINING:
                return False
            view = self._maintenance_view_locked()
            if (
                not view["processes_verified"]
                or view["slots_in_use"] != 0
                or view["unsettled_inference"] != 0
                or view["unsettled_swaps"] != 0
            ):
                return False
            now = time.time()
            self._state["maintenance_phase"] = MAINTENANCE_STOPPED
            self._state["maintenance_stopped_at"] = now
            self._state["maintenance_changed_at"] = now
            return True

    def fail_maintenance(self, token: str, generation: int, error: str) -> bool:
        """Record a failed stop.  Never reports the machine as serviceable."""
        with self._lock:
            if not self._maintenance_token_matches_locked(token, generation):
                return False
            if self._state["maintenance_phase"] != MAINTENANCE_DRAINING:
                return False
            now = time.time()
            self._state["maintenance_phase"] = MAINTENANCE_FAILED
            self._state["maintenance_error"] = error
            self._state["maintenance_verified"] = False
            self._state["maintenance_failed_at"] = now
            self._state["maintenance_changed_at"] = now
            return True

    def resume_maintenance(self) -> int:
        """Reopen admission.  Only valid from ``stopped`` or ``failed``.

        Bumps the generation so any waiter from an earlier stop can detect that
        its outcome is stale and must not report the machine as serviceable.

        The active version is always cleared, so inference keeps returning 503
        until a new upload is promoted.  If a *failed* stop left a worker
        process alive, its version is moved to ``retired``/``draining`` so the
        drain loop stops it once any accepted work genuinely finishes -- resume
        never kills unsettled inference and never resumes serving on a process
        the stop was trying to remove.
        """
        with self._lock:
            phase = self._state["maintenance_phase"]
            if phase == MAINTENANCE_RUNNING:
                raise MaintenanceConflict("the service is not in maintenance")
            if phase == MAINTENANCE_DRAINING:
                raise MaintenanceConflict(
                    "a maintenance shutdown is still draining; wait for it to settle"
                )
            now = time.time()
            active = self._state["active_version"]
            if active is not None:
                info = self._versions.get(active)
                if info is not None:
                    info = dict(info)
                    info.update(
                        state=VERSION_RETIRED,
                        process_state=STATE_DRAINING,
                        retired_at=now,
                    )
                    self._versions[active] = info
                    slot = self._slots.get(info["slot_id"])
                    if slot is not None and slot["version"] == active:
                        slot = dict(slot)
                        slot["process_state"] = STATE_DRAINING
                        self._slots[info["slot_id"]] = slot
            self._state["active_version"] = None
            generation = int(self._state["maintenance_generation"]) + 1
            self._state["maintenance_generation"] = generation
            self._state["maintenance_phase"] = MAINTENANCE_RUNNING
            self._state["maintenance_token"] = None
            self._state["maintenance_verified"] = False
            self._state["maintenance_changed_at"] = now
            return generation

    def maintenance_outcome(self, token: str, generation: int) -> Dict[str, Any]:
        """Return the outcome *as seen for the caller's generation*.

        ``stale`` is true whenever the generation moved on (a resume or a new
        stop), in which case ``safe_to_service`` is forced to false: a waiter
        that began before an interleaving may never certify the machine.
        """
        with self._lock:
            current = int(self._state["maintenance_generation"])
            view = self._maintenance_view_locked()
            stale = generation != current or self._state["maintenance_token"] != token
            view["stale"] = stale
            view["waiter_generation"] = generation
            if stale:
                view["safe_to_service"] = False
            return view

    def release_after_stop(
        self, slot_id: str, version: str, process_exited: bool
    ) -> bool:
        """Free a slot during maintenance once its OS process is proven gone.

        Unlike ``release_slot`` this accepts an ``active`` or ``failed`` slot;
        it still refuses while any accepted work is counted in flight, so a
        slot can never look free while its process is still computing.
        """
        if not process_exited:
            return False
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None or slot["version"] != version:
                return False
            if slot["in_flight"] != 0:
                return False
            info = self._versions.get(version)
            now = time.time()
            slot = dict(slot)
            slot.update(
                version=None,
                process_state=STATE_IDLE,
                pid=None,
                in_flight=0,
                idle_since=now,
            )
            self._slots[slot_id] = slot
            if info is not None and info.get("slot_id") == slot_id:
                info = dict(info)
                if info["state"] != VERSION_FAILED:
                    info["state"] = VERSION_STOPPED
                info["process_state"] = STATE_IDLE
                info["stopped_at"] = now
                self._versions[version] = info
            if self._state["active_version"] == version:
                self._state["active_version"] = None
            return True

    def _maintenance_token_matches_locked(self, token: str, generation: int) -> bool:
        return (
            int(self._state["maintenance_generation"]) == generation
            and self._state["maintenance_token"] == token
        )

    # ------------------------------------------------------------------
    # Admission / registration
    # ------------------------------------------------------------------
    def begin_swap(
        self,
        expected_version: Optional[str],
        token: str,
        enforce_expected: bool = True,
    ) -> Optional[str]:
        """Claim the right to replace the current active version.

        ``expected_version`` is the version the caller observed as active when
        it decided to swap.  If ``enforce_expected`` is set and the active
        version has changed since then, the caller is stale and gets
        ``SwapConflict``.  The claim is keyed by the current old version, so of
        several swaps that observed the *same* old version at most one can
        proceed.
        """
        with self._lock:
            self._require_admission_locked("model replacement")
            active = self._state["active_version"]
            if enforce_expected and expected_version != active:
                raise SwapConflict(
                    f"swap precondition failed: caller observed active version "
                    f"{expected_version!r}, but the current active version is {active!r}"
                )
            key = active if active is not None else CLAIM_NONE
            if key in self._claims:
                raise SwapConflict(
                    f"a swap from version {active!r} is already in progress"
                )
            self._claims[key] = token
            return active

    def end_swap(self, old_version: Optional[str], token: str) -> None:
        key = old_version if old_version is not None else CLAIM_NONE
        with self._lock:
            if self._claims.get(key) == token:
                del self._claims[key]

    def reserve_any_slot(self, version: str) -> str:
        """Atomically pick an idle slot and register a loading candidate.

        A version id whose previous attempt *failed* or was *stopped* for
        maintenance may be uploaded again after resume; the new attempt keeps
        the earlier error for auditing.  Any live version id is rejected, so a
        serving or draining version can never be overwritten.
        """
        with self._lock:
            previous = self._versions.get(version)
            attempt = 1
            previous_error = None
            if previous is not None:
                if previous["state"] not in (VERSION_FAILED, VERSION_STOPPED):
                    raise SwapConflict(
                        f"version {version!r} already exists ({previous['state']})"
                    )
                attempt = int(previous.get("attempt", 1)) + 1
                previous_error = previous.get("error")
            free = sorted(
                sid
                for sid, s in self._slots.items()
                if s["process_state"] == STATE_IDLE
            )
            if not free:
                raise NoCapacity(
                    "both model process slots are occupied (loading/active/"
                    "draining); wait for the draining version to finish"
                )
            slot_id = free[0]
            now = time.time()
            slot = dict(self._slots[slot_id])
            slot.update(
                version=version,
                process_state=STATE_LOADING,
                pid=None,
                in_flight=0,
                accepted=0,
                completed=0,
                started_at=now,
                idle_since=None,
                generation=slot["generation"] + 1,
            )
            self._slots[slot_id] = slot
            self._versions[version] = {
                "version": version,
                "slot_id": slot_id,
                "pid": None,
                "state": VERSION_LOADING,
                "process_state": STATE_LOADING,
                "created_at": now,
                "tested_at": None,
                "activated_at": None,
                "retired_at": None,
                "error": None,
                "attempt": attempt,
                "previous_error": previous_error,
            }
            return slot_id

    def set_pid(self, version: str, pid: int) -> None:
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                return
            info = dict(info)
            info["pid"] = pid
            self._versions[version] = info
            slot = self._slots.get(info["slot_id"])
            if slot is not None and slot["version"] == version:
                slot = dict(slot)
                slot["pid"] = pid
                self._slots[info["slot_id"]] = slot

    def set_model_shape(self, version: str, n_features: int, n_classes: int) -> None:
        """Record the candidate's input/output dimensions for request validation."""
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                return
            info = dict(info)
            info["n_features"] = int(n_features)
            info["n_classes"] = int(n_classes)
            self._versions[version] = info

    def mark_candidate_tested(self, version: str) -> None:
        with self._lock:
            info = self._versions.get(version)
            if info is None or info["state"] != VERSION_LOADING:
                raise SwapConflict(f"version {version!r} is not a loading candidate")
            info = dict(info)
            info["tested_at"] = time.time()
            self._versions[version] = info

    def fail_candidate(self, version: str, error: str) -> None:
        """Mark a loading candidate failed and free its slot (process exited)."""
        with self._lock:
            info = self._versions.get(version)
            if info is None or info["state"] != VERSION_LOADING:
                return
            info = dict(info)
            info.update(
                state=VERSION_FAILED,
                process_state=STATE_IDLE,
                error=error,
                retired_at=time.time(),
            )
            self._versions[version] = info
            slot = self._slots.get(info["slot_id"])
            if slot is not None and slot["version"] == version:
                slot = dict(slot)
                slot.update(
                    version=None,
                    process_state=STATE_IDLE,
                    pid=None,
                    in_flight=0,
                    idle_since=time.time(),
                )
                self._slots[info["slot_id"]] = slot

    # ------------------------------------------------------------------
    # Commit / inference / drain
    # ------------------------------------------------------------------
    def commit_swap(self, version: str, old_version: Optional[str], token: str) -> None:
        with self._lock:
            active = self._state["active_version"]
            if active != old_version:
                raise SwapConflict(
                    f"active version changed from {old_version!r} to {active!r}; "
                    "candidate was not promoted"
                )
            key = active if active is not None else CLAIM_NONE
            if self._claims.get(key) != token:
                raise SwapConflict("swap claim was lost")
            info = self._versions.get(version)
            if info is None or info["state"] != VERSION_LOADING:
                raise SwapConflict(f"version {version!r} is not a loading candidate")
            if not info.get("tested_at"):
                raise SwapConflict(f"version {version!r} has not passed its smoke test")

            if active is not None:
                old = dict(self._versions[active])
                old.update(
                    state=VERSION_RETIRED,
                    process_state=STATE_DRAINING,
                    retired_at=time.time(),
                )
                self._versions[active] = old
                old_slot = self._slots.get(old["slot_id"])
                if old_slot is not None and old_slot["version"] == active:
                    old_slot = dict(old_slot)
                    old_slot["process_state"] = STATE_DRAINING
                    self._slots[old["slot_id"]] = old_slot

            new = dict(info)
            new.update(
                state=VERSION_ACTIVE,
                process_state=STATE_ACTIVE,
                activated_at=time.time(),
            )
            self._versions[version] = new
            slot = self._slots.get(new["slot_id"])
            if slot is None or slot["version"] != version:
                raise SwapConflict("candidate slot ownership changed")
            slot = dict(slot)
            slot["process_state"] = STATE_ACTIVE
            self._slots[new["slot_id"]] = slot
            self._state["active_version"] = version
            del self._claims[key]

    def accept_work(self, n_features: Optional[int] = None) -> Tuple[str, str]:
        """Pin an inference to the active version and count it in-flight.

        With ``n_features`` the request is validated against the active
        version's input width inside the same critical section, so a rejected
        request never touches the in-flight counter.
        """
        with self._lock:
            self._require_admission_locked("inference")
            version = self._state["active_version"]
            if version is None:
                raise NoActiveModel("no model is active")
            info = self._versions.get(version)
            if info is None or info["state"] != VERSION_ACTIVE:
                raise NoActiveModel(f"version {version!r} is not active")
            if (
                n_features is not None
                and info.get("n_features") is not None
                and int(info["n_features"]) != n_features
            ):
                raise FeatureMismatch(
                    f"version {version!r} expects {info['n_features']} features, "
                    f"got {n_features}",
                    version=version,
                    expected=int(info["n_features"]),
                )
            slot = self._slots.get(info["slot_id"])
            if (
                slot is None
                or slot["version"] != version
                or slot["process_state"] != STATE_ACTIVE
            ):
                raise NoActiveModel("active slot is inconsistent")
            slot = dict(slot)
            slot["in_flight"] += 1
            slot["accepted"] += 1
            self._slots[info["slot_id"]] = slot
            return version, info["slot_id"]

    def complete_work(self, version: str) -> None:
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                return
            slot = self._slots.get(info["slot_id"])
            if slot is None or slot["version"] != version:
                return
            slot = dict(slot)
            slot["in_flight"] = max(0, slot["in_flight"] - 1)
            slot["completed"] += 1
            self._slots[info["slot_id"]] = slot

    def release_slot(self, slot_id: str, version: str, process_exited: bool) -> bool:
        """Free a drained slot.

        ``process_exited`` must be True: slot reuse is gated on the OS process
        actually being gone, not on the API's intent to stop it.
        """
        if not process_exited:
            return False
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None or slot["version"] != version:
                return False
            if slot["in_flight"] != 0:
                return False
            if slot["process_state"] not in (STATE_DRAINING, STATE_LOADING):
                return False
            slot = dict(slot)
            slot.update(
                version=None,
                process_state=STATE_IDLE,
                pid=None,
                idle_since=time.time(),
            )
            self._slots[slot_id] = slot
            info = self._versions.get(version)
            if info is not None:
                info = dict(info)
                info["process_state"] = STATE_IDLE
                self._versions[version] = info
            return True

    def fail_active(self, version: str, error: str) -> bool:
        """Take a crashed active version out of service and free its slot.

        Used when the worker process dies unexpectedly.  Refuses while work is
        still counted in-flight (those requests will fail on their own and
        decrement the counter), and never touches a different active version.
        """
        with self._lock:
            info = self._versions.get(version)
            if info is None or info["state"] != VERSION_ACTIVE:
                return False
            slot = self._slots.get(info["slot_id"])
            if slot is None or slot["version"] != version or slot["in_flight"] != 0:
                return False
            info = dict(info)
            info.update(state=VERSION_FAILED, error=error, retired_at=time.time())
            self._versions[version] = info
            slot = dict(slot)
            slot.update(
                version=None,
                process_state=STATE_IDLE,
                pid=None,
                in_flight=0,
                idle_since=time.time(),
            )
            self._slots[info["slot_id"]] = slot
            if self._state["active_version"] == version:
                self._state["active_version"] = None
            return True

    def settle_dead_worker(self, version: str) -> bool:
        """If a worker died mid-request, stop counting its in-flight work.

        Called by the drain/reaper loop once the OS process is gone: the
        corresponding HTTP calls have failed or will fail, so keeping the
        counter raised would wedge the slot forever.
        """
        with self._lock:
            info = self._versions.get(version)
            if info is None:
                return False
            slot = self._slots.get(info["slot_id"])
            if slot is None or slot["version"] != version:
                return False
            if slot["in_flight"] == 0:
                return False
            slot = dict(slot)
            slot["completed"] += slot["in_flight"]
            slot["in_flight"] = 0
            self._slots[info["slot_id"]] = slot
            return True


# ---------------------------------------------------------------------------
# API-side worker process handle
# ---------------------------------------------------------------------------


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


class WorkerHandle:
    """One model worker OS process and its private control directory."""

    def __init__(
        self,
        slot_id: str,
        version: str,
        model_path: str,
        work_dir: str,
        host: str = "127.0.0.1",
    ) -> None:
        self.slot_id = slot_id
        self.version = version
        self.model_path = model_path
        self.host = host
        self.port: Optional[int] = None
        self.pid: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self.control_dir = os.path.join(work_dir, f"{slot_id}-{uuid.uuid4().hex[:8]}")
        os.makedirs(self.control_dir, exist_ok=True)
        self.ready_file = os.path.join(self.control_dir, "ready.json")

    def inference_url(self) -> str:
        if self.port is None:
            raise RuntimeError(f"worker {self.version} is not listening yet")
        return f"http://{self.host}:{self.port}/infer"

    def status_url(self) -> str:
        if self.port is None:
            raise RuntimeError(f"worker {self.version} is not listening yet")
        return f"http://{self.host}:{self.port}/status"

    def spawn(
        self,
        slow_load_ms: int = 0,
        simulate_load_failure: bool = False,
        load_timeout: float = 30.0,
    ) -> None:
        """Start the worker process (does not wait for readiness)."""
        root = str(Path(__file__).resolve().parent.parent)
        cmd = [
            sys.executable,
            "-m",
            "app.worker_launcher",
            self.slot_id,
            self.version,
            self.model_path,
            self.control_dir,
            self.ready_file,
            self.host,
            str(slow_load_ms),
            "1" if simulate_load_failure else "0",
        ]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self._proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self.pid = self._proc.pid

    def wait_ready(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Block until the worker writes its ready handshake or exits."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready = _read_json(self.ready_file)
            if ready is not None:
                if ready.get("ok") and ready.get("port"):
                    self.port = int(ready["port"])
                return ready
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"worker exited during load (exit code {self._proc.returncode})"
                )
            time.sleep(0.02)
        raise TimeoutError(f"worker did not become ready within {timeout:.1f}s")

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def exited(self) -> bool:
        """True only once the OS process is provably gone.

        With a ``Popen`` handle this means the child has been reaped (no
        zombie left); for a handle without a ``Popen`` it falls back to a
        signal-0 liveness probe on the recorded pid.
        """
        if self._proc is not None:
            return self._proc.poll() is not None
        if self.pid is None:
            return True
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def wait_exit(self, timeout: float) -> bool:
        if self._proc is None:
            return True
        try:
            self._proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate the process group and reap the child."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc.wait()
            return
        self._signal_group(signal.SIGTERM)
        if not self.wait_exit(timeout):
            self._signal_group(signal.SIGKILL)
            self.wait_exit(timeout)

    def _signal_group(self, sig: int) -> None:
        try:
            os.killpg(os.getpgid(self.pid), sig)  # type: ignore[arg-type]
        except (ProcessLookupError, PermissionError, TypeError):
            pass
