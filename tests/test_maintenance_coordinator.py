"""In-process tests for stop-coordinator failure paths.

These use the real ``InferenceService`` + ``Registry`` but replace worker
handles with fakes, so we can exercise "process will not die" and "slot still
occupied" without needing an unkillable real process.  The invariant under test
is absolute: a stop that cannot prove all processes exited must report failure,
never ``safe_to_service``.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os

import pytest

from app import main as api_main
from app.process_manager import Registry


class FakeHandle:
    """A worker handle whose exit can be scripted."""

    def __init__(self, version, slot_id, *, exit_on_stop=True):
        self.version = version
        self.slot_id = slot_id
        self.pid = None  # no real process -> pid_alive() is False
        self.stop_calls = 0
        self._exit_on_stop = exit_on_stop
        self._exited = False

    def alive(self):
        return not self._exited

    def exited(self):
        return self._exited

    def stop(self, timeout=10.0):
        self.stop_calls += 1
        if self._exit_on_stop:
            self._exited = True

    def wait_exit(self, timeout=10.0):
        return self._exited


@pytest.fixture()
def service(monkeypatch):
    ctx = multiprocessing.get_context("fork")
    mgr = ctx.Manager()
    svc = api_main.InferenceService()
    svc.manager = mgr
    svc.registry = Registry(mgr)
    svc.client = None
    svc._proc_lock = asyncio.Lock()
    svc._stop_lock = asyncio.Lock()
    monkeypatch.setattr(api_main, "MAINTENANCE_SETTLE_TIMEOUT_SECONDS", 2.0)
    yield svc
    mgr.shutdown()


def _serve(reg: Registry, version: str, token_old="__bootstrap__") -> str:
    slot = reg.reserve_any_slot(version)
    reg.set_pid(version, -1)  # -1 => not a real live pid
    reg.begin_swap(None, token_old)
    reg.mark_candidate_tested(version)
    reg.commit_swap(version, None, token_old)
    return slot


def test_stop_fails_when_process_cannot_be_proven_dead(service):
    reg = service.registry
    slot = _serve(reg, "v1")
    service.handles["v1"] = FakeHandle("v1", slot, exit_on_stop=False)

    async def run():
        token, generation = await service.request_stop("test")
        view = await service.wait_for_stop(token, generation, 5.0)
        return token, generation, view

    token, generation, view = asyncio.run(run())
    assert view["phase"] == "failed"
    assert view["safe_to_service"] is False
    assert view["error"]
    assert view["slots_in_use"] >= 1
    assert service.handles["v1"].stop_calls == 1

    # Resume is allowed from failed, but must not serve the still-live worker.
    reg.resume_maintenance()
    assert reg.active_version() is None
    assert reg.version_info("v1")["state"] == "retired"
    assert asyncio.run(service.wait_for_stop(token, generation, 0.0))["stale"] is True


def test_stop_marks_failed_when_settle_times_out(service):
    reg = service.registry
    slot = _serve(reg, "v1")
    service.handles["v1"] = FakeHandle("v1", slot, exit_on_stop=True)
    reg.accept_work()  # pretend accepted inference never comes back

    async def run():
        token, generation = await service.request_stop("test")
        return token, generation, await service.wait_for_stop(token, generation, 10.0)

    token, generation, view = asyncio.run(run())
    assert view["phase"] == "failed"
    assert "timed out" in (view["error"] or "")
    assert view["safe_to_service"] is False
    # The process was never touched while real work was outstanding.
    assert service.handles["v1"].stop_calls == 0
    assert service.handles["v1"].alive() is True
    reg.complete_work("v1")


def test_stop_succeeds_and_zeroes_slots_when_handle_dies(service):
    reg = service.registry
    slot = _serve(reg, "v1")
    service.handles["v1"] = FakeHandle("v1", slot, exit_on_stop=True)

    async def run():
        token, generation = await service.request_stop("test")
        return await service.wait_for_stop(token, generation, 5.0)

    view = asyncio.run(run())
    assert view["phase"] == "stopped"
    assert view["safe_to_service"] is True
    assert view["processes_verified"] is True
    assert view["slots_in_use"] == 0
    assert all(s["version"] is None for s in reg.snapshot()["slots"])
    assert reg.active_version() is None
    assert service.handles == {}


def test_concurrent_stop_requests_share_one_generation(service):
    reg = service.registry
    slot = _serve(reg, "v1")
    handle = FakeHandle("v1", slot, exit_on_stop=True)
    service.handles["v1"] = handle

    async def run():
        a = await service.request_stop("first")
        b = await service.request_stop("second")
        va = await service.wait_for_stop(a[0], a[1], 5.0)
        vb = await service.wait_for_stop(b[0], b[1], 5.0)
        return a, b, va, vb

    a, b, va, vb = asyncio.run(run())
    assert a == b  # same token + generation; no orphaned first waiter
    assert va["safe_to_service"] is True
    assert vb["safe_to_service"] is True
    assert handle.stop_calls == 1  # process stopped exactly once
    assert service.handles == {}


class PidOnlyHandle(FakeHandle):
    """Handle with no Popen but a recorded pid, to exercise the pid probe."""

    def __init__(self, version, slot_id, pid):
        super().__init__(version, slot_id, exit_on_stop=True)
        self.pid = pid

    def alive(self):
        return True  # look inconsistent on purpose; exited() must decide


def test_stop_reclaims_slot_with_dead_recorded_pid(service):
    """A slot whose worker died and whose handle vanished must be reclaimable."""
    reg = service.registry
    _serve(reg, "v1")
    # No handle at all, pid -1 => provably not alive (hidden dead-worker path).
    reg.set_pid("v1", -1)
    service.handles.clear()

    async def run():
        token, generation = await service.request_stop("test")
        return await service.wait_for_stop(token, generation, 5.0)

    view = asyncio.run(run())
    assert view["phase"] == "stopped"
    assert view["safe_to_service"] is True
    assert view["slots_in_use"] == 0


def test_stop_refuses_slot_with_live_pid_and_no_handle(service):
    """A recorded live pid with no handle: must not be certified safe."""
    reg = service.registry
    _serve(reg, "v1")
    reg.set_pid("v1", os.getpid())  # our own pid: certainly alive
    service.handles.clear()

    async def run():
        token, generation = await service.request_stop("test")
        return await service.wait_for_stop(token, generation, 5.0)

    view = asyncio.run(run())
    assert view["phase"] == "failed"
    assert view["safe_to_service"] is False
    assert "live pid" in (view["error"] or "")
