"""Checkpoint watcher stays device-free until actual work is available."""
from supermimic.evaluation_queue import pending_checkpoints, wait_for_checkpoint


def test_only_eligible_unfinished_checkpoints(tmp_path):
    for it in (500, 1000, 1500, 2000, 3000):
        (tmp_path / f"ckpt_{it:06d}.pt").touch()
    result = pending_checkpoints(tmp_path, {1000}, iteration_interval=1000, stop_after=2000)
    assert result == [str(tmp_path / 'ckpt_002000.pt')]
    assert pending_checkpoints(tmp_path, set(), every=2) == [
        str(tmp_path / f"ckpt_{it:06d}.pt") for it in (500, 1000, 1500)]


def test_wait_handles_checkpoint_arrival_without_simulator(tmp_path, monkeypatch):
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        (tmp_path / 'ckpt_002000.pt').touch()
    monkeypatch.setattr('supermimic.evaluation_queue.time.sleep', sleep)
    (tmp_path / 'ckpt_001500.pt').touch()
    assert wait_for_checkpoint(tmp_path, iteration_interval=1000, poll_seconds=.1)
    assert sleeps == [.1]


def test_stopped_failed_and_completed_runs_do_not_initialize(tmp_path):
    for marker in ('STOPPED.json', 'invalid_transition.pt'):
        p = tmp_path / marker; p.touch()
        assert not wait_for_checkpoint(tmp_path)
        p.unlink()
    (tmp_path / 'eval.csv').write_text('iteration,n_failed\n2000,0\n')
    assert not wait_for_checkpoint(tmp_path, stop_after=2000)
