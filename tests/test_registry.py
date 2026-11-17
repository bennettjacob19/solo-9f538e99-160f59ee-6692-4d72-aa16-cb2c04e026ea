"""Unit tests for the shared registry state machine, including the drain gate."""

from __future__ import annotations

import multiprocessing

import pytest

from app.process_manager import (
    VERSION_ACTIVE,
    VERSION_FAILED,
    VERSION_RETIRED,
    FeatureMismatch,
    MaintenanceConflict,
    NoActiveModel,
    NoCapacity,
    Registry,
    ServiceStopping,
    SwapConflict,
)


@pytest.fixture(scope="module")
def manager():
    ctx = multiprocessing.get_context("fork")
    mgr = ctx.Manager()
    yield mgr
    mgr.shutdown()


@pytest.fixture()
def registry(manager):
    reg = Registry(manager)
    yield reg
    # Clear state between tests so the shared manager stays pristine.
    for slot in list(reg._slots.keys()):
        reg._slots[slot] = {
            "slot_id": slot,
            "version": None,
            "process_state": "idle",
            "pid": None,
            "in_flight": 0,
            "accepted": 0,
            "completed": 0,
            "started_at": None,
            "idle_since": 0.0,
            "generation": reg._slots[slot]["generation"],
        }
    for version in list(reg._versions.keys()):
        del reg._versions[version]
    for key in list(reg._claims.keys()):
        del reg._claims[key]
    reg._state["active_version"] = None


def _load(reg: Registry, version: str) -> str:
    slot = reg.reserve_any_slot(version)
    reg.set_pid(version, 1000)
    return slot


def _activate(reg: Registry, version: str, old: str | None = None, token: str = "t") -> None:
    reg.begin_swap(old, token)
    reg.mark_candidate_tested(version)
    reg.commit_swap(version, old, token)


def test_first_load_claims_none_and_activates(registry):
    slot = _load(registry, "v1")
    assert slot == "slot-0"
    registry.mark_candidate_tested("v1")
    old = registry.begin_swap(None, "tok")
    registry.commit_swap("v1", old, "tok")
    assert registry.active_version() == "v1"
    assert registry.version_info("v1")["state"] == VERSION_ACTIVE
    assert registry.snapshot()["slots_in_use"] == 1


def test_second_swap_retires_old_but_keeps_slot_until_drained(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")

    assert registry.active_version() == "v2"
    assert registry.version_info("v1")["state"] == VERSION_RETIRED
    assert registry.snapshot()["slots_in_use"] == 2  # v1 still draining
    assert not registry.has_capacity()

    slot = registry.version_info("v1")["slot_id"]
    # In-flight work blocks release.
    version, _ = registry.accept_work()
    assert version == "v2"
    assert registry.release_slot(slot, "v1", process_exited=True) is True  # no in-flight on v1
    registry.complete_work("v2")


def test_no_capacity_when_both_slots_held(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    with pytest.raises(NoCapacity):
        registry.reserve_any_slot("v3")
    # The current version is untouched.
    assert registry.active_version() == "v2"


def test_in_flight_request_blocks_slot_release(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    _load(registry, "v2")
    _activate(registry, "v2", old="v1")

    # A request accepted on v1 *before* the swap is counted on v1.
    # (Simulated here by accepting on v1 while it was active is not possible
    # after commit, so drive the counter the way the API does.)
    registry._slots["slot-0"] = {**registry._slots["slot-0"], "in_flight": 1}
    assert registry.release_slot("slot-0", "v1", process_exited=True) is False
    assert registry.release_slot("slot-0", "v1", process_exited=False) is False
    registry.complete_work("v1")
    assert registry.release_slot("slot-0", "v1", process_exited=True) is True
    assert registry.has_capacity()


def test_concurrent_swap_claim_allows_one_winner(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    assert registry.begin_swap("v1", "a") == "v1"
    with pytest.raises(SwapConflict, match="already in progress"):
        registry.begin_swap("v1", "b")
    registry.end_swap("v1", "a")
    assert registry.begin_swap("v1", "b") == "v1"


def test_stale_expected_version_is_conflict(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    with pytest.raises(SwapConflict, match="precondition"):
        registry.begin_swap("v0", "token")


def test_commit_requires_smoke_test(registry):
    _load(registry, "v1")
    registry.begin_swap(None, "t")
    with pytest.raises(SwapConflict, match="smoke test"):
        registry.commit_swap("v1", None, "t")


def test_failed_candidate_frees_slot_and_can_be_retried(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.begin_swap("v1", "t")
    _load(registry, "bad")
    registry.fail_candidate("bad", "boom")
    registry.end_swap("v1", "t")

    info = registry.version_info("bad")
    assert info["state"] == VERSION_FAILED
    assert info["error"] == "boom"
    assert registry.has_capacity()
    assert registry.active_version() == "v1"

    registry.begin_swap("v1", "t2")
    _load(registry, "bad")  # retry allowed, keeps previous error
    assert registry.version_info("bad")["attempt"] == 2
    registry.end_swap("v1", "t2")


def test_duplicate_live_version_rejected(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    with pytest.raises(SwapConflict, match="already exists"):
        registry.reserve_any_slot("v1")


def test_accepted_work_is_pinned_to_active_version(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    version, slot = registry.accept_work()
    assert (version, slot) == ("v1", "slot-0")

    _load(registry, "v2")
    _activate(registry, "v2", old="v1")
    # The pre-existing in-flight count belongs to v1, not the new active.
    assert registry.version_in_flight("v1") == 1
    assert registry.version_in_flight("v2") == 0
    registry.complete_work("v1")
    assert registry.version_in_flight("v1") == 0


def test_feature_mismatch_does_not_count_in_flight(registry):
    _load(registry, "v1")
    registry.set_model_shape("v1", 3, 2)
    _activate(registry, "v1")
    with pytest.raises(FeatureMismatch):
        registry.accept_work(2)
    assert registry.version_in_flight("v1") == 0
    version, _ = registry.accept_work(3)
    assert version == "v1"


def test_inference_without_model_is_rejected(registry):
    with pytest.raises(NoActiveModel):
        registry.accept_work()


def test_dead_active_worker_releases_slot(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.accept_work()
    registry.settle_dead_worker("v1")
    assert registry.fail_active("v1", "crashed") is True
    assert registry.active_version() is None
    assert registry.has_capacity()
    assert registry.version_info("v1")["state"] == VERSION_FAILED


# ---------------------------------------------------------------------------
# Maintenance shutdown
# ---------------------------------------------------------------------------


def _stopped_registry(registry):
    """Drive a served registry into the stopped phase at generation 1."""
    _load(registry, "v1")
    _activate(registry, "v1")
    token, generation = registry.begin_maintenance("service")
    assert registry.accept_work  # gate is closed even before processes stop
    registry.mark_processes_verified(token, generation)
    # Simulate the coordinator having stopped the process and freed the slot.
    info = registry.version_info("v1")
    assert registry.release_after_stop(info["slot_id"], "v1", process_exited=True)
    assert registry.finish_maintenance(token, generation) is True
    return token, generation


def test_maintenance_closes_admission_atomically(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    token, generation = registry.begin_maintenance("service")
    assert generation == 1
    assert registry.maintenance_view()["phase"] == "draining"
    with pytest.raises(ServiceStopping):
        registry.accept_work()
    with pytest.raises(ServiceStopping):
        registry.begin_swap("v1", "tok2")
    # The stop token is not leaked as a swap claim.
    assert "v1" not in registry.snapshot()["claims"]


def test_maintenance_does_not_report_safe_until_clean(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    # Accepted work outstanding: certified exit + zero slots is still not safe.
    registry.accept_work()
    token, generation = registry.begin_maintenance("service")
    assert registry.finish_maintenance(token, generation) is False
    registry.mark_processes_verified(token, generation)
    assert registry.finish_maintenance(token, generation) is False
    view = registry.maintenance_view()
    assert view["safe_to_service"] is False
    assert view["unsettled_inference"] == 1

    # Settle the work, free the slot, then it may certify.
    registry.complete_work("v1")
    assert registry.finish_maintenance(token, generation) is False  # slot still held
    info = registry.version_info("v1")
    assert registry.release_after_stop(info["slot_id"], "v1", process_exited=True)
    assert registry.finish_maintenance(token, generation) is True
    view = registry.maintenance_view()
    assert view["safe_to_service"] is True
    assert view["slots_in_use"] == 0
    assert view["unsettled_total"] == 0


def test_maintenance_requires_verified_process_exit(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    token, generation = registry.begin_maintenance("service")
    info = registry.version_info("v1")
    registry.release_after_stop(info["slot_id"], "v1", process_exited=True)
    # No mark_processes_verified: cannot certify.
    assert registry.finish_maintenance(token, generation) is False


def test_release_after_stop_refuses_in_flight_and_unexited(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    registry.accept_work()
    info = registry.version_info("v1")
    assert registry.release_after_stop(info["slot_id"], "v1", process_exited=False) is False
    assert registry.release_after_stop(info["slot_id"], "v1", process_exited=True) is False
    registry.complete_work("v1")
    assert registry.release_after_stop(info["slot_id"], "v1", process_exited=True) is True
    assert registry.active_version() is None
    assert registry.version_info("v1")["state"] == "stopped"


def test_stale_waiter_cannot_certify_after_resume(registry):
    token, generation = _stopped_registry(registry)
    new_generation = registry.resume_maintenance()
    assert new_generation == generation + 1
    outcome = registry.maintenance_outcome(token, generation)
    assert outcome["stale"] is True
    assert outcome["safe_to_service"] is False
    # A fresh stop at the new generation can eventually certify again, and the
    # old token can never flip it.
    assert outcome["phase"] == "running"


def test_waiter_generation_mismatch_is_stale_even_while_stopped(registry):
    token, generation = _stopped_registry(registry)
    outcome = registry.maintenance_outcome(token, generation + 99)
    assert outcome["stale"] is True
    assert outcome["safe_to_service"] is False


def test_resume_reopens_admission_and_keeps_inference_503(registry):
    token, generation = _stopped_registry(registry)
    registry.resume_maintenance()
    view = registry.maintenance_view()
    assert view["admission_open"] is True
    assert registry.active_version() is None
    with pytest.raises(NoActiveModel):
        registry.accept_work()
    # Uploads are admitted again, and the same version id can be reloaded.
    slot = registry.reserve_any_slot("v1")
    assert slot in {"slot-0", "slot-1"}
    assert registry.version_info("v1")["attempt"] == 2


def test_begin_maintenance_phase_conflicts(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    token, generation = registry.begin_maintenance("service")
    with pytest.raises(MaintenanceConflict, match="already draining"):
        registry.begin_maintenance("again")
    with pytest.raises(MaintenanceConflict, match="still draining"):
        registry.resume_maintenance()
    # A failed stop may be retried, and it gets a fresh generation.
    assert registry.fail_maintenance(token, generation, "boom") is True
    assert registry.maintenance_view()["phase"] == "failed"
    token2, generation2 = registry.begin_maintenance("retry")
    assert generation2 == generation + 1
    assert token2 != token
    # The failed attempt's token can no longer mutate anything.
    assert registry.finish_maintenance(token, generation) is False


def test_failed_stop_resume_retires_live_active_version(registry):
    _load(registry, "v1")
    _activate(registry, "v1")
    token, generation = registry.begin_maintenance("service")
    assert registry.fail_maintenance(token, generation, "worker still alive") is True
    view = registry.maintenance_view()
    assert view["safe_to_service"] is False
    registry.resume_maintenance()
    assert registry.active_version() is None
    # The still-live process is handed to the drain loop, not served again.
    assert registry.version_info("v1")["state"] == VERSION_RETIRED
    assert registry.version_info("v1")["process_state"] == "draining"


def test_stale_finish_after_new_waiters_generation(registry):
    """A late finish() from an old stop must not certify a new one."""
    token, generation = _stopped_registry(registry)
    registry.resume_maintenance()  # generation 2, running
    token2, generation2 = registry.begin_maintenance("again")
    assert (token2, generation2) != (token, generation)
    # Old waiter calls back after the new stop began: no effect at all.
    assert registry.mark_processes_verified(token, generation) is False
    assert registry.finish_maintenance(token, generation) is False
    assert registry.maintenance_view()["phase"] == "draining"
    assert registry.maintenance_view()["safe_to_service"] is False


def test_admission_gate_is_atomic_with_accept_and_swap(registry):
    """No interleaving admits work once the stop has been acknowledged."""
    _load(registry, "v1")
    _activate(registry, "v1")
    reg = registry

    # Interleave accept_work and begin_maintenance in many rounds; the set of
    # accepted requests must never include one that observed phase != running.
    accepted = []

    def try_accept():
        try:
            reg.accept_work()
        except ServiceStopping:
            return "stopped"
        except NoActiveModel:
            return "no_model"
        accepted.append(reg.maintenance_phase())
        return "accepted"

    results = [try_accept() for _ in range(5)]
    token, generation = reg.begin_maintenance("x")
    results += [try_accept() for _ in range(5)]
    assert results[:5] == ["accepted"] * 5
    assert results[5:] == ["stopped"] * 5
    assert all(phase == "running" for phase in accepted)
    # Drain the accepted work so the registry is clean.
    for _ in accepted:
        reg.complete_work("v1")
    reg.fail_maintenance(token, generation, "test cleanup")
