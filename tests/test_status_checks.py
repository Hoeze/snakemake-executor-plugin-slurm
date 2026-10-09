import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from snakemake_executor_plugin_slurm import Executor


class _NoopAsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_executor(status_command, init_seconds, max_seconds, next_seconds):
    executor = Executor.__new__(Executor)
    executor.logger = MagicMock()
    executor.run_uuid = "run-uuid"
    executor.status_rate_limiter = _NoopAsyncContext()
    executor.get_status_command = lambda: status_command
    executor.next_seconds_between_status_checks = next_seconds
    executor._status_query_calls = 0
    executor._status_query_failures = 0
    executor._status_query_total_seconds = 0.0
    executor._status_query_min_seconds = None
    executor._status_query_max_seconds = 0.0
    executor._status_query_cycle_rows = []
    executor._preemption_warning = False
    executor._failed_nodes = set()
    executor.report_job_success = MagicMock()
    executor.report_job_error = MagicMock()
    executor.workflow = SimpleNamespace(
        executor_settings=SimpleNamespace(
            status_attempts=1,
            init_seconds_before_status_checks=init_seconds,
            max_seconds_between_status_checks=max_seconds,
            keep_successful_logs=True,
            requeue=False,
        )
    )
    return executor


def _mock_status_queries(monkeypatch, squeue_output, sacct_output):
    """Mock squeue and sacct output; return the list of issued commands."""
    commands = []

    async def _mock_query_job_status(command, logger):
        commands.append(command)
        if command == "squeue":
            return (squeue_output, 0.01)
        return (sacct_output, 0.01)

    def _mock_sacct_command(run_uuid, job_ids=None):
        if job_ids:
            return f"sacct --jobs {','.join(job_ids)}"
        return "sacct"

    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm.query_job_status",
        _mock_query_job_status,
    )
    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm.query_job_status_squeue",
        lambda run_uuid: "squeue",
    )
    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm.query_job_status_sacct",
        _mock_sacct_command,
    )
    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm.get_min_job_age",
        lambda: 60,
    )
    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm.is_query_tool_available",
        lambda tool: True,
    )
    return commands


def _make_job(external_jobid):
    return SimpleNamespace(
        external_jobid=external_jobid,
        aux={"slurm_logfile": Path(f"{external_jobid}.log")},
    )


def _check_active_jobs(executor, active_jobs):
    async def _collect_remaining_jobs():
        remaining = []
        async for job in executor.check_active_jobs(active_jobs):
            remaining.append(job)
        return remaining

    return asyncio.run(_collect_remaining_jobs())


@pytest.mark.parametrize(
    "init_seconds, max_seconds, next_seconds, expected_next_seconds",
    [
        # without finished jobs, the interval grows by 10 s up to the maximum
        (40, 60, 55, 60),
        # init == max gives a fixed interval
        (30, 30, 30, 30),
    ],
)
def test_status_interval_is_capped_by_max_setting(
    monkeypatch, init_seconds, max_seconds, next_seconds, expected_next_seconds
):
    executor = _make_executor("sacct", init_seconds, max_seconds, next_seconds)
    _mock_status_queries(monkeypatch, {}, {"101": "RUNNING"})
    job = _make_job("101")

    remaining = _check_active_jobs(executor, [job])

    assert remaining == [job]
    assert executor.next_seconds_between_status_checks == expected_next_seconds
