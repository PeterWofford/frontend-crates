# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Determinism, failure propagation, and generation ownership across process workers."""
import json
import multiprocessing
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import generation_workers


def _job(value):
    if value == "fail":
        raise ValueError("worker rejected input")
    return value, generation_workers.worker_count()


def _blocking_job(item):
    value, release = item
    if value == 0:
        assert release.wait(5), "completed work was blocked behind the first submitted job"
    return value


@pytest.fixture
def tiny_worker_admission(monkeypatch):
    # Pool semantics need their requested participants even on a constrained host;
    # the real CPU, cgroup, and memory admission policy has its own tests.
    def admitted_count(jobs=None):
        if multiprocessing.current_process().name != "MainProcess":
            return 1
        requested = generation_workers._requested.get()
        count = 2 if requested == "auto" else int(requested)
        return max(1, min(count, jobs or count))

    monkeypatch.setattr(generation_workers, "worker_count", admitted_count)


def test_completion_drain_releases_fast_jobs_without_submission_order_blocking(tiny_worker_admission):
    with multiprocessing.Manager() as manager, generation_workers.worker_policy(2):
        release = manager.Event()
        results = generation_workers.bounded_map(_blocking_job, [(0, release), (1, release)], phase="completion-test", ordered=False)
        try:
            assert next(results) == 1
            release.set()
            assert list(results) == [0]
        finally:
            release.set()
            results.close()


def test_worker_admission_honors_affinity_memory_and_nested_process(monkeypatch):
    with generation_workers.worker_policy("auto"):
        monkeypatch.setattr(generation_workers.os, "sched_getaffinity", lambda pid: {0})
        assert generation_workers.worker_count() == 1
        monkeypatch.setattr(generation_workers.os, "sched_getaffinity", lambda pid: set(range(32)))
        monkeypatch.setattr(generation_workers, "_available_memory", lambda: 1)
        assert generation_workers.worker_count() == 1
        monkeypatch.setattr(generation_workers.multiprocessing, "current_process", lambda: SimpleNamespace(name="worker"))
        assert generation_workers.worker_count() == 1


def test_final_serial_job_cannot_escape_memory_guard(monkeypatch):
    def crosses_limit(value):
        monkeypatch.setattr(generation_workers, "MEMORY_LIMIT", 1)
        return value

    with generation_workers.worker_policy(1), pytest.raises(MemoryError, match="64 GiB"):
        list(generation_workers.bounded_map(crosses_limit, [1], phase="memory-test"))


def test_pool_orders_results_prevents_nested_pools_and_propagates_failure(tiny_worker_admission):
    with generation_workers.worker_policy(2):
        assert list(generation_workers.bounded_map(_job, [3, 1, 2], phase="test")) == [(3, 1), (1, 1), (2, 1)]
        with pytest.raises(ValueError, match="worker rejected input"):
            list(generation_workers.bounded_map(_job, [3, "fail", 2], phase="test"))
