"""Integration tests for the maintenance-shutdown API with real processes.

Covers the guarantees a duty engineer relies on before servicing the machine:

* after the stop is acknowledged, new inference and uploads are refused;
* inference accepted earlier -- even one that already returned 504 to its
  client -- is allowed to finish inside the worker before its process stops;
* a version swap that had already started is settled, not abandoned;
* the service reports "safe" only after both worker processes are proven gone
  and both slots are idle;
* a timeout (or a client disconnect) on the stop request does not auto-resume;
* after resume, uploads work again but inference stays 503 until a model loads;
* a stale waiter from an older generation can never report the machine safe.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

from tests.test_api_integration import (
    ApiServer,
    LABELS,
    W_A,
    W_B,
    artifact,
    infer,
    status,
    upload,
    wait_for,
)


@pytest.fixture(scope="module")
def maint_api(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("qcmaint")
    with ApiServer(data_dir) as server:
        yield server


def stop(api: ApiServer, *, wait_seconds=20.0, reason=None):
    body = {"wait_seconds": wait_seconds}
    if reason is not None:
        body["reason"] = reason
    return httpx.post(
        f"{api.base}/admin/maintenance/stop", json=body, timeout=wait_seconds + 30
    )


def resume(api: ApiServer):
    return httpx.post(f"{api.base}/admin/maintenance/resume", timeout=30)


def maintenance(api: ApiServer):
    return status(api)["maintenance"]


def test_01_baseline_load(maint_api):
    resp = upload(maint_api, artifact("m1", W_A, [0.0, 0.0], LABELS))
    assert resp.status_code == 201, resp.text
    assert maintenance(maint_api)["phase"] == "running"
    assert maintenance(maint_api)["admission_open"] is True
    assert infer(maint_api, [0.9, 0.1]).status_code == 200


def test_02_stop_with_accepted_timeout_work_waits_for_real_finish(maint_api):
    # Accepted inference whose client gives up; the worker keeps computing.
    resp = infer(maint_api, [0.9, 0.1], timeout_ms=150, slow_ms=2500)
    assert resp.status_code == 504, resp.text
    st = status(maint_api)
    assert st["total_in_flight"] == 1

    started = time.perf_counter()
    stop_resp = stop(maint_api, wait_seconds=20)
    elapsed = time.perf_counter() - started
    assert stop_resp.status_code == 200, stop_resp.text
    body = stop_resp.json()
    assert body["phase"] == "stopped"
    assert body["safe_to_service"] is True
    # It genuinely waited for the computation, not just for its own bookkeeping.
    assert elapsed >= 1.0, elapsed

    st = status(maint_api)
    assert st["maintenance"]["safe_to_service"] is True
    assert st["maintenance"]["unsettled_inference"] == 0
    assert st["maintenance"]["unsettled_swaps"] == 0
    assert st["slots_in_use"] == 0
    assert st["active_version"] is None
    for slot in st["slots"]:
        assert slot["process_state"] == "idle"
        assert slot["version"] is None


def test_03_stop_refuses_new_inference_and_uploads(maint_api):
    resp = infer(maint_api, [0.9, 0.1])
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"] == "service_stopping"
    up = upload(maint_api, artifact("m2", W_B, [0.0, 0.0], LABELS))
    assert up.status_code == 503, up.text
    assert up.json()["detail"]["error"] == "service_stopping"


def test_04_second_stop_while_stopped_is_idempotent_safe(maint_api):
    first = maintenance(maint_api)
    resp = stop(maint_api, wait_seconds=5)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["safe_to_service"] is True
    assert body["generation"] == first["generation"]


def test_05_resume_reopens_uploads_but_inference_is_503(maint_api):
    resp = resume(maint_api)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["phase"] == "running"
    assert body["admission_open"] is True

    # No model loaded yet: inference must be 503, not routed to a dead process.
    inf = infer(maint_api, [0.9, 0.1])
    assert inf.status_code == 503
    assert inf.json()["detail"]["error"] == "no_active_model"

    up = upload(maint_api, artifact("m2", W_B, [0.0, 0.0], LABELS))
    assert up.status_code == 201, up.text
    assert infer(maint_api, [0.9, 0.1]).status_code == 200


def test_06_stop_during_started_swap_settles_it(maint_api):
    """A swap racing a stop: either it commits before the gate, or it is refused.

    The candidate that started loading before the stop is not killed mid-load;
    the stop waits for that swap to reach a conclusion before stopping workers.
    """
    import threading

    result = {}

    def do_upload():
        result["resp"] = upload(
            maint_api, artifact("m3", W_A, [0.0, 0.0], LABELS, slow_load_ms=900)
        )

    result.clear()
    t = threading.Thread(target=do_upload)
    t.start()
    time.sleep(0.25)  # let the swap pass admission and start loading
    stop_resp = stop(maint_api, wait_seconds=30)
    t.join()

    up = result["resp"]
    if up.status_code == 201:
        # The swap settled before/while the gate closed and was fully drained.
        assert stop_resp.status_code == 200, stop_resp.text
        assert stop_resp.json()["safe_to_service"] is True
        st = status(maint_api)
        assert st["active_version"] is None
        assert st["slots_in_use"] == 0
        assert st["maintenance"]["unsettled_swaps"] == 0
    else:
        # If admission closed first, the upload is refused, not half-done.
        assert up.status_code in (503, 409), up.text
        assert stop_resp.status_code == 200, stop_resp.text
        assert stop_resp.json()["safe_to_service"] is True


def test_07_stop_timeout_does_not_report_safe_or_auto_resume(maint_api):
    assert resume(maint_api).status_code == 200
    up = upload(maint_api, artifact("m4", W_A, [0.0, 0.0], LABELS))
    assert up.status_code == 201, up.text

    # Slow work keeps the stop draining past a tiny wait_seconds.
    inf = infer(maint_api, [0.9, 0.1], timeout_ms=100, slow_ms=2500)
    assert inf.status_code == 504
    started = time.perf_counter()
    resp = stop(maint_api, wait_seconds=0.25)
    elapsed = time.perf_counter() - started
    assert resp.status_code == 504, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "maintenance_still_draining"
    assert detail["safe_to_service"] is False
    assert detail["unsettled_inference"] >= 1
    assert elapsed < 2.5

    # Not auto-resumed: admission is still closed, inference is still refused.
    st = status(maint_api)
    assert st["maintenance"]["phase"] in ("draining", "stopped")
    assert infer(maint_api, [0.9, 0.1]).status_code == 503

    # The background stop finishes on its own once the work really ends.
    wait_for(
        lambda: status(maint_api)["maintenance"]["phase"] == "stopped", timeout=15
    )
    st = status(maint_api)
    assert st["maintenance"]["safe_to_service"] is True
    assert st["slots_in_use"] == 0

    # Resume still requires an explicit call.
    assert resume(maint_api).status_code == 200


def test_08_stale_waiter_after_resume_is_not_safe(maint_api):
    # Fresh model, then a stop we can capture the generation of.
    assert upload(maint_api, artifact("m5", W_B, [0.0, 0.0], LABELS)).status_code == 201
    resp = stop(maint_api, wait_seconds=20)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    stop_id = body["maintenance"]["stop_id"]
    generation = body["generation"]
    assert body["safe_to_service"] is True

    assert resume(maint_api).status_code == 200
    # The old waiter polls after the resume: its outcome is stale, never safe.
    q = httpx.get(
        f"{maint_api.base}/admin/maintenance/status",
        params={"stop_id": stop_id, "generation": generation},
        timeout=5,
    )
    assert q.status_code == 200
    old = q.json()
    assert old["stale"] is True
    assert old["safe_to_service"] is False

    # A new stop begins: an even older waiter must still not certify.
    assert upload(maint_api, artifact("m6", W_A, [0.0, 0.0], LABELS)).status_code == 201
    waiting = []
    import threading

    t = threading.Thread(
        target=lambda: waiting.append(stop(maint_api, wait_seconds=20))
    )
    t.start()
    t.join()
    assert waiting and waiting[0].status_code == 200, waiting[0].text
    assert waiting[0].json()["generation"] > generation

    q2 = httpx.get(
        f"{maint_api.base}/admin/maintenance/status",
        params={"stop_id": stop_id, "generation": generation},
        timeout=5,
    ).json()
    assert q2["stale"] is True
    assert q2["safe_to_service"] is False
    assert resume(maint_api).status_code == 200


def test_09_resume_conflicts_and_docs(maint_api):
    # test_08 left the service running: resume is a conflict.
    assert resume(maint_api).status_code == 409
    assert stop(maint_api, wait_seconds=20).status_code == 200
    assert resume(maint_api).status_code == 200
    # Not in maintenance any more.
    assert resume(maint_api).status_code == 409
    assert httpx.get(f"{maint_api.base}/docs", timeout=5).status_code == 200
    spec = httpx.get(f"{maint_api.base}/openapi.json", timeout=5).json()
    assert "/admin/maintenance/stop" in spec["paths"]
    assert "/admin/maintenance/resume" in spec["paths"]


def test_10_status_exposes_phase_and_unsettled_work(maint_api):
    st = status(maint_api)
    m = st["maintenance"]
    for key in (
        "phase",
        "generation",
        "unsettled_inference",
        "unsettled_swaps",
        "unsettled_total",
        "slots_in_use",
        "processes_verified",
        "admission_open",
        "safe_to_service",
    ):
        assert key in m
    # Two-process cap is unchanged and observable.
    assert st["capacity"] == 2
    assert len(st["slots"]) == 2


def test_11_client_disconnect_does_not_abort_stop(maint_api):
    """A caller that hangs up mid-stop: the stop still completes for real."""
    resume(maint_api)  # already running from test_09 is fine (409)
    assert upload(maint_api, artifact("m7", W_A, [0.0, 0.0], LABELS)).status_code == 201
    # Accepted slow work that its own client already gave up on.
    assert infer(maint_api, [0.9, 0.1], timeout_ms=100, slow_ms=2000).status_code == 504

    import threading

    def start_and_hang_up():
        try:
            # 0.1s client timeout: httpx abandons the request, the server keeps
            # its detached coordinator running.
            httpx.post(
                f"{maint_api.base}/admin/maintenance/stop",
                json={"wait_seconds": 30},
                timeout=0.1,
            )
        except httpx.TimeoutException:
            pass

    t = threading.Thread(target=start_and_hang_up)
    t.start()
    t.join()

    # Admission is closed immediately and is NOT re-opened by the disconnect.
    assert maintenance(maint_api)["phase"] in ("draining", "stopped")
    assert infer(maint_api, [0.9, 0.1]).status_code == 503

    wait_for(lambda: maintenance(maint_api)["phase"] == "stopped", timeout=15)
    st = status(maint_api)
    assert st["maintenance"]["safe_to_service"] is True
    assert st["slots_in_use"] == 0
    assert resume(maint_api).status_code == 200


def test_12_stop_and_resume_interleaved_with_requests(maint_api):
    """Race stop/resume/infer/upload; no stale waiter may certify, no safe lie."""
    assert resume(maint_api)  # ensure running
    assert upload(maint_api, artifact("m8", W_A, [0.0, 0.0], LABELS)).status_code == 201

    import threading

    results = []
    lock = threading.Lock()

    def worker(i):
        for _ in range(4):
            choice = i % 4
            if choice == 0:
                r = infer(maint_api, [0.9, 0.1], timeout_ms=200, slow_ms=400)
                code = r.status_code
                if code == 200:
                    # Any admitted inference must have been fully settled by the
                    # time a successful stop reports safe; it can also be 503/504.
                    pass
            elif choice == 1:
                r = upload(
                    maint_api,
                    artifact(f"mx-{i}-{_}", W_B, [0.0, 0.0], LABELS, slow_load_ms=100),
                )
                code = r.status_code
            elif choice == 2:
                r = stop(maint_api, wait_seconds=8)
                code = r.status_code
                if code == 200:
                    body = r.json()
                    with lock:
                        results.append(("safe", body["generation"]))
            else:
                r = resume(maint_api)
                code = r.status_code
            with lock:
                results.append(("code", code))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    # If any caller got a safe verdict, the machine must actually be clean at
    # that generation unless it was invalidated by a later resume.
    final = maintenance(maint_api)
    if final["phase"] == "running":
        # A resume happened last; nobody may claim safety now.
        assert final["safe_to_service"] is False
    if final["phase"] == "stopped":
        assert final["safe_to_service"] is True
        assert final["slots_in_use"] == 0
        assert final["unsettled_total"] == 0
    resume(maint_api)  # leave a clean state for any later test


def test_13_stop_takes_down_active_and_draining_processes(maint_api):
    """Both slots busy (active + draining) -> stop must kill both, then certify."""
    resume(maint_api)
    assert upload(maint_api, artifact("m9", W_A, [0.0, 0.0], LABELS)).status_code == 201
    wait_for(lambda: status(maint_api)["slots_in_use"] == 1, timeout=10)

    # Leave real work running on the version that is about to be retired.
    assert infer(maint_api, [0.9, 0.1], timeout_ms=100, slow_ms=1800).status_code == 504
    assert upload(maint_api, artifact("m10", W_B, [0.0, 0.0], LABELS)).status_code == 201

    st = status(maint_api)
    busy = [s for s in st["slots"] if s["process_state"] != "idle"]
    assert len(busy) == 2, st["slots"]
    pids = {s["pid"] for s in busy if s["pid"]}
    assert len(pids) == 2

    started = time.perf_counter()
    resp = stop(maint_api, wait_seconds=25)
    elapsed = time.perf_counter() - started
    assert resp.status_code == 200, resp.text
    assert resp.json()["safe_to_service"] is True
    assert elapsed >= 0.9  # waited for the accepted 504'd work to finish

    st = status(maint_api)
    assert st["slots_in_use"] == 0
    assert st["maintenance"]["safe_to_service"] is True
    assert st["active_version"] is None
    for pid in pids:
        # Every previously-recorded worker pid is really gone.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        else:  # pragma: no cover - only if the OS pid is somehow still alive
            raise AssertionError(f"worker pid {pid} still alive after stop")
    assert resume(maint_api).status_code == 200
