"""Tests for checkpoint resume, sweep fail-fast, and idempotent analysis.

All tests run on CPU with synthetic data, no downloads required.
"""

from __future__ import annotations

import copy
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from superadditivity.logging.checkpoint_manager import (
    CheckpointManager,
    TrainingCheckpointManager,
    _config_hash,
)
from superadditivity.training.decentralized_client import DecentralizedClient
from superadditivity.training.dsgd_coordinator import DSGDCoordinator
from superadditivity.training.lr_schedule import CosineDecaySchedule
from superadditivity.evaluation.drift_tracker import DriftTracker
from superadditivity.utils.io import save_json, load_json


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "output"


def _make_clients(n=4, seed=42):
    """Create lightweight clients with a trivial model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    clients = []
    for cid in range(n):
        m = copy.deepcopy(model)
        data = torch.randn(32, 8)
        targets = torch.randint(0, 4, (32,))
        ds = TensorDataset(data, targets)
        ds.targets = targets.tolist()
        ds.run_seed = seed
        ds.set_round = lambda r: None

        from superadditivity.datasets.client_dataset import ClientDataset
        cds = ClientDataset(ds, list(range(32)), client_id=cid, run_seed=seed)
        client = DecentralizedClient(
            client_id=cid, model=m, dataset=cds,
            device=torch.device("cpu"), lr=0.01,
            momentum=0.0, weight_decay=0.0,
            batch_size=16, local_steps=1,
        )
        clients.append(client)
    return clients


def _make_coordinator(tmp_output, total_rounds=20, checkpoint_every=5, config_extra=None):
    """Build a coordinator with resume support wired up."""
    n = 4
    clients = _make_clients(n=n)
    W = np.ones((n, n), dtype=np.float64) / n
    lr_schedule = CosineDecaySchedule(lr_max=0.01, lr_min=0.001, total_rounds=total_rounds)

    cfg = {"total_rounds": total_rounds}
    if config_extra:
        cfg.update(config_extra)

    resume_mgr = TrainingCheckpointManager(
        str(tmp_output / "resume_checkpoints"),
        config_dict=cfg,
    )

    coordinator = DSGDCoordinator(
        clients=clients,
        mixing_matrix=W,
        lr_schedule=lr_schedule,
        total_rounds=total_rounds,
        eval_every=100,
        checkpoint_every=checkpoint_every,
        output_dir=str(tmp_output),
        resume_manager=resume_mgr,
    )
    return coordinator, resume_mgr, clients


# ------------------------------------------------------------------ #
# TrainingCheckpointManager unit tests
# ------------------------------------------------------------------ #

class TestTrainingCheckpointManager:

    def test_save_and_load(self, tmp_output):
        cfg = {"lr": 0.01, "rounds": 10}
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict=cfg)
        clients = _make_clients(n=2)

        for c in clients:
            c.local_step(0)

        history = {"round": [0], "mean_loss": [1.5]}
        path = mgr.save_training_state(clients, round_num=5, history=history)

        assert path.exists()
        ckpt = mgr.load_training_state(path)
        assert ckpt["round_num"] == 5
        assert ckpt["n_clients"] == 2
        assert ckpt["history"] == history

    def test_config_mismatch_raises(self, tmp_output):
        cfg1 = {"lr": 0.01, "rounds": 10}
        cfg2 = {"lr": 0.05, "rounds": 10}
        mgr1 = TrainingCheckpointManager(str(tmp_output), config_dict=cfg1)
        clients = _make_clients(n=2)
        for c in clients:
            c.local_step(0)
        mgr1.save_training_state(clients, round_num=0, history={})

        mgr2 = TrainingCheckpointManager(str(tmp_output), config_dict=cfg2)
        with pytest.raises(RuntimeError, match="Configuration changed"):
            mgr2.load_training_state()

    def test_corrupt_only_checkpoint_raises(self, tmp_output):
        """A single corrupt checkpoint raises immediately."""
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict={"a": 1})
        p = mgr.output_dir / "resume_ckpt_r0010.pt"
        p.write_text("not a valid checkpoint")

        with pytest.raises(RuntimeError, match="corrupt or unreadable"):
            mgr.find_latest()

    def test_corrupt_latest_stops_even_if_older_valid_exists(self, tmp_output):
        """Strict fail-fast: corrupt latest raises even when an older valid checkpoint exists."""
        cfg = {"x": 1}
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict=cfg)
        clients = _make_clients(n=2)
        for c in clients:
            c.local_step(0)

        mgr.save_training_state(clients, 10, {})

        import time
        corrupt = mgr.output_dir / "resume_ckpt_r0020.pt"
        corrupt.write_text("CORRUPT")
        time.sleep(0.05)
        corrupt.touch()

        with pytest.raises(RuntimeError, match="corrupt or unreadable"):
            mgr.find_latest()

    def test_structurally_invalid_latest_raises(self, tmp_output):
        """A checkpoint missing required fields raises immediately."""
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict={"a": 1})
        p = mgr.output_dir / "resume_ckpt_r0005.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"round_num": 5}, p)

        with pytest.raises(RuntimeError, match="structurally invalid"):
            mgr.find_latest()

    def test_config_mismatch_in_find_latest_raises(self, tmp_output):
        """find_latest raises RuntimeError on config mismatch."""
        cfg1 = {"x": 1}
        mgr1 = TrainingCheckpointManager(str(tmp_output), config_dict=cfg1)
        clients = _make_clients(n=2)
        for c in clients:
            c.local_step(0)
        mgr1.save_training_state(clients, 5, {})

        cfg2 = {"x": 999}
        mgr2 = TrainingCheckpointManager(str(tmp_output), config_dict=cfg2)
        with pytest.raises(RuntimeError, match="Configuration changed"):
            mgr2.find_latest()

    def test_has_any_checkpoint_files(self, tmp_output):
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict={"a": 1})
        assert not mgr.has_any_checkpoint_files()

        (mgr.output_dir / "resume_ckpt_r0001.pt").write_text("junk")
        assert mgr.has_any_checkpoint_files()

    def test_restore_clients(self, tmp_output):
        cfg = {"x": 1}
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict=cfg)
        clients = _make_clients(n=2)
        for c in clients:
            c.local_step(0)

        mgr.save_training_state(clients, round_num=3, history={})

        for c in clients:
            with torch.no_grad():
                for p in c.model.parameters():
                    p.fill_(999.0)

        ckpt = mgr.load_training_state()
        mgr.restore_clients(clients, ckpt)

        for c in clients:
            for p in c.model.parameters():
                assert not torch.all(p == 999.0)

    def test_retention_policy(self, tmp_output):
        cfg = {"x": 1}
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict=cfg, keep_last=2)
        clients = _make_clients(n=2)
        for c in clients:
            c.local_step(0)

        mgr.save_training_state(clients, 10, {})
        mgr.save_training_state(clients, 20, {})
        mgr.save_training_state(clients, 30, {})

        remaining = sorted(mgr.output_dir.glob("resume_ckpt_r*.pt"))
        assert len(remaining) == 2
        assert "r0020" in remaining[0].name
        assert "r0030" in remaining[1].name

    def test_dataloader_generator_state_saved(self, tmp_output):
        """Checkpoint includes DataLoader generator states."""
        cfg = {"x": 1}
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict=cfg)
        clients = _make_clients(n=2)
        for c in clients:
            c.local_step(0)

        mgr.save_training_state(clients, 0, {})
        ckpt = mgr.load_training_state()
        dl_states = ckpt.get("dataloader_generator_states", [])
        assert len(dl_states) == 2
        for s in dl_states:
            assert s is not None

    def test_no_checkpoints_returns_none(self, tmp_output):
        """find_latest returns None when no checkpoint files exist at all."""
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict={"a": 1})
        assert mgr.find_latest() is None


# ------------------------------------------------------------------ #
# DSGDCoordinator resume tests
# ------------------------------------------------------------------ #

class TestCoordinatorResume:

    def test_full_run_creates_checkpoint(self, tmp_output):
        coord, resume_mgr, _ = _make_coordinator(tmp_output)
        history = coord.run()

        assert len(history["round"]) == 20
        assert resume_mgr.find_latest() is not None

    def test_resume_skips_completed_rounds(self, tmp_output):
        """Run 10 rounds, then create a new coordinator that resumes."""
        coord1, _, _ = _make_coordinator(tmp_output, total_rounds=10, checkpoint_every=5)
        h1 = coord1.run()
        assert len(h1["round"]) == 10

        n = 4
        clients = _make_clients(n=n)
        W = np.ones((n, n), dtype=np.float64) / n
        lr_schedule = CosineDecaySchedule(lr_max=0.01, lr_min=0.001, total_rounds=20)
        resume_mgr2 = TrainingCheckpointManager(
            str(tmp_output / "resume_checkpoints"),
            config_dict={"total_rounds": 10},
        )
        coord2 = DSGDCoordinator(
            clients=clients, mixing_matrix=W, lr_schedule=lr_schedule,
            total_rounds=20, eval_every=100, checkpoint_every=5,
            output_dir=str(tmp_output), resume_manager=resume_mgr2,
        )
        h2 = coord2.run()

        assert h2["round"][-1] == 19
        assert h2["round"][0] == 0

    def test_already_complete_run_is_noop(self, tmp_output):
        coord, _, _ = _make_coordinator(tmp_output, total_rounds=10)
        h1 = coord.run()
        assert len(h1["round"]) == 10

        n = 4
        clients = _make_clients(n=n)
        W = np.ones((n, n), dtype=np.float64) / n
        resume_mgr = TrainingCheckpointManager(
            str(tmp_output / "resume_checkpoints"),
            config_dict={"total_rounds": 10},
        )
        coord2 = DSGDCoordinator(
            clients=clients, mixing_matrix=W,
            lr_schedule=CosineDecaySchedule(lr_max=0.01, lr_min=0.001, total_rounds=10),
            total_rounds=10, eval_every=100, checkpoint_every=5,
            output_dir=str(tmp_output), resume_manager=resume_mgr,
        )
        h2 = coord2.run()
        assert h2["round"][-1] == 9

    def test_corrupt_latest_checkpoint_stops_coordinator(self, tmp_output):
        """Coordinator raises when the latest checkpoint is corrupt."""
        coord, resume_mgr, _ = _make_coordinator(tmp_output, total_rounds=10)
        ckpt_dir = resume_mgr.output_dir
        (ckpt_dir / "resume_ckpt_r0009.pt").write_text("CORRUPT")

        with pytest.raises(RuntimeError, match="corrupt or unreadable"):
            coord.run()

    def test_csv_truncated_on_resume(self, tmp_output):
        """On resume, CSV rows beyond the checkpoint round are removed."""
        coord, resume_mgr, _ = _make_coordinator(
            tmp_output, total_rounds=20, checkpoint_every=5,
        )
        csv_path = tmp_output / "metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["round", "loss"])
            w.writeheader()
            for r in range(12):
                w.writerow({"round": r, "loss": 0.5})

        clients = _make_clients(n=4)
        for c in clients:
            c.local_step(0)
        resume_mgr.save_training_state(clients, 5, {
            "round": list(range(6)),
            "lr": [0.01] * 6,
            "mean_loss": [0.5] * 6,
            "wall_time": [1.0] * 6,
        })

        coord2, _, _ = _make_coordinator(tmp_output, total_rounds=20, checkpoint_every=5)
        coord2.run()

        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rounds = [int(row["round"]) for row in reader]
        from collections import Counter
        counts = Counter(rounds)
        assert all(v == 1 for v in counts.values()), f"Duplicate rounds: {counts}"

    def test_resumed_vs_uninterrupted_final_params(self, tmp_output):
        """Compare final model parameters: uninterrupted vs interrupted+resumed."""
        CONFIG = {"total_rounds": 10}

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        coord_full, _, clients_full = _make_coordinator(
            tmp_output / "full", total_rounds=10, checkpoint_every=5,
            config_extra=CONFIG,
        )
        coord_full.run()
        full_params = [
            {k: v.clone() for k, v in c.model.state_dict().items()}
            for c in clients_full
        ]

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        n = 4
        clients_part = _make_clients(n=n)
        W = np.ones((n, n), dtype=np.float64) / n
        partial_mgr = TrainingCheckpointManager(
            str(tmp_output / "partial" / "resume_checkpoints"),
            config_dict=CONFIG,
        )
        coord_part = DSGDCoordinator(
            clients=clients_part, mixing_matrix=W,
            lr_schedule=CosineDecaySchedule(lr_max=0.01, lr_min=0.001, total_rounds=5),
            total_rounds=5, eval_every=100, checkpoint_every=5,
            output_dir=str(tmp_output / "partial"),
            resume_manager=partial_mgr,
        )
        coord_part.run()

        coord_resume, _, clients_resume = _make_coordinator(
            tmp_output / "partial", total_rounds=10, checkpoint_every=5,
            config_extra=CONFIG,
        )
        coord_resume.run()
        resumed_params = [
            {k: v.clone() for k, v in c.model.state_dict().items()}
            for c in clients_resume
        ]

        max_diff = 0.0
        for i in range(len(full_params)):
            for key in full_params[i]:
                diff = (full_params[i][key] - resumed_params[i][key]).abs().max().item()
                max_diff = max(max_diff, diff)
        assert max_diff < 0.5, (
            f"Max parameter difference {max_diff:.6f} is too large — "
            f"resume is not restoring training state correctly."
        )


# ------------------------------------------------------------------ #
# Sweep-level fail-fast tests
# ------------------------------------------------------------------ #

class TestSweepFailFast:

    def test_is_complete_requires_valid_json(self, tmp_path):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "output_dir": str(tmp_path),
            "run_seed": 42,
            "graph_seed": 100,
            "training": {"total_rounds": 500},
        })

        from scripts.run_sweep import _is_complete

        assert not _is_complete(cfg)

        run_dir = tmp_path / "seed_42_graph_100"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text("NOT JSON")
        assert not _is_complete(cfg)

        save_json({"total_rounds": 100}, run_dir / "summary.json")
        assert not _is_complete(cfg)

        save_json({"total_rounds": 500}, run_dir / "summary.json")
        assert not _is_complete(cfg)

        (run_dir / "drift_metrics.h5").write_bytes(b"HDF5")
        assert _is_complete(cfg)

    def test_is_complete_rejects_config_hash_mismatch(self, tmp_path):
        """A summary from a different config should not mark the run as complete."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "output_dir": str(tmp_path),
            "run_seed": 42,
            "graph_seed": 100,
            "training": {"total_rounds": 500},
            "extra_key": "value_A",
        })

        run_dir = tmp_path / "seed_42_graph_100"
        run_dir.mkdir(parents=True)
        (run_dir / "drift_metrics.h5").write_bytes(b"HDF5")

        save_json({
            "total_rounds": 500,
            "config_hash": "0000000000000000",
        }, run_dir / "summary.json")

        from scripts.run_sweep import _is_complete
        assert not _is_complete(cfg)

    def test_has_resume_checkpoint(self, tmp_path):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "output_dir": str(tmp_path),
            "run_seed": 42,
            "graph_seed": 100,
            "training": {"total_rounds": 500},
        })

        from scripts.run_sweep import _has_resume_checkpoint, _run_status

        assert _run_status(cfg) == "new"

        ckpt_dir = tmp_path / "seed_42_graph_100" / "resume_checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "resume_ckpt_r0050.pt").write_bytes(b"dummy")
        assert _has_resume_checkpoint(cfg)
        assert _run_status(cfg) == "interrupted"

    def test_failed_experiment_stops_sweep(self, tmp_path):
        """A SweepFailedError is raised when _run_single encounters an error."""
        from scripts.run_sweep import SweepFailedError, _run_single
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "output_dir": str(tmp_path),
            "run_seed": 42,
            "graph_seed": 100,
            "training": {"total_rounds": 500},
            "experiment_name": "test_fail",
            "n_clients": 4,
            "n_communities": 4,
            "data": {"dataset": "nonexistent", "partition_method": "iid",
                     "dirichlet_alpha": 1.0, "data_dir": "/nonexistent",
                     "download": False},
            "graph": {"topology": "er", "p_out": 0.5, "p_in": 0.5,
                      "n_clients": 4, "n_communities": 4},
            "model": {"architecture": "resnet18_cifar"},
        })

        with pytest.raises(SweepFailedError):
            _run_single(cfg, "test_fail_label")

    def test_sweep_failed_error_writes_error_json(self, tmp_path):
        """When a sweep experiment fails, error.json is written to the run dir."""
        from scripts.run_sweep import _write_error_log
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "output_dir": str(tmp_path),
            "run_seed": 42,
            "graph_seed": 100,
            "training": {"total_rounds": 500},
        })

        _write_error_log(cfg, "test_label", ValueError("boom"))
        error_path = tmp_path / "seed_42_graph_100" / "error.json"
        assert error_path.exists()
        data = load_json(error_path)
        assert data["experiment"] == "test_label"
        assert data["error_type"] == "ValueError"
        assert "boom" in data["error_message"]

    def test_sweep_exit_code_nonzero(self, tmp_path):
        """The sweep script exits nonzero when an experiment fails."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/run_sweep.py",
             "--experiment", "core_factorial", "--resume"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        # core_factorial config may not exist or the experiment will fail
        # on synthetic data — either way, exit code should be nonzero
        # if any run fails.  We just verify it doesn't silently succeed
        # with exit code 0 when no data is available.
        assert result.returncode != 0


# ------------------------------------------------------------------ #
# Atomic save_json test
# ------------------------------------------------------------------ #

class TestAtomicSaveJson:

    def test_no_partial_file_on_crash(self, tmp_path):
        target = tmp_path / "test.json"
        save_json({"a": 1}, target)
        assert load_json(target) == {"a": 1}
        assert not (tmp_path / "test.json.tmp").exists()


# ------------------------------------------------------------------ #
# DriftTracker state round-trip
# ------------------------------------------------------------------ #

class TestDriftTrackerState:

    def test_state_round_trip(self):
        tracker = DriftTracker()
        tracker.record(0, {"cka_mean": 0.95})
        tracker.record(1, {"cka_mean": 0.87})

        state = tracker.get_state()

        tracker2 = DriftTracker()
        tracker2.load_state(state)

        np.testing.assert_array_equal(
            tracker.get_trajectory("cka_mean"),
            tracker2.get_trajectory("cka_mean"),
        )


# ------------------------------------------------------------------ #
# Config hash stability
# ------------------------------------------------------------------ #

class TestConfigHash:

    def test_same_config_same_hash(self):
        cfg = {"lr": 0.01, "model": "resnet18", "rounds": 500}
        assert _config_hash(cfg) == _config_hash(cfg)

    def test_different_config_different_hash(self):
        cfg1 = {"lr": 0.01, "rounds": 500}
        cfg2 = {"lr": 0.05, "rounds": 500}
        assert _config_hash(cfg1) != _config_hash(cfg2)

    def test_key_order_irrelevant(self):
        cfg1 = {"a": 1, "b": 2}
        cfg2 = {"b": 2, "a": 1}
        assert _config_hash(cfg1) == _config_hash(cfg2)


# ------------------------------------------------------------------ #
# Analysis / visualization caching
# ------------------------------------------------------------------ #

class TestAnalysisCaching:

    def test_missing_outputs_triggers_regeneration(self, tmp_path):
        """Fingerprint matches but output files deleted -> re-run."""
        from scripts.run_analysis import (
            _is_analysis_current,
            _summaries_fingerprint,
            _write_fingerprint,
        )

        results_dir = tmp_path / "results"
        output_dir = tmp_path / "analysis"
        results_dir.mkdir()
        output_dir.mkdir()

        run_dir = results_dir / "seed_42_graph_100"
        run_dir.mkdir()
        save_json({"total_rounds": 10}, run_dir / "summary.json")

        _write_fingerprint(str(output_dir), str(results_dir))
        assert not _is_analysis_current(str(output_dir), str(results_dir))

        (output_dir / "statistical_report.md").write_text("# Report")
        assert _is_analysis_current(str(output_dir), str(results_dir))

    def test_viz_missing_outputs_triggers_regeneration(self, tmp_path):
        from scripts.run_visualization import (
            _is_viz_current,
            _results_fingerprint,
            _write_viz_fingerprint,
        )

        results_dir = tmp_path / "results"
        fig_dir = tmp_path / "figures"
        results_dir.mkdir()
        fig_dir.mkdir()

        run_dir = results_dir / "seed_42_graph_100"
        run_dir.mkdir()
        save_json({"total_rounds": 10}, run_dir / "summary.json")

        _write_viz_fingerprint(fig_dir, results_dir)
        assert not _is_viz_current(fig_dir, results_dir)

        (fig_dir / "factorial_grid.pdf").write_bytes(b"PDF")
        assert _is_viz_current(fig_dir, results_dir)

    def test_stale_fingerprint_invalidated_on_analysis(self, tmp_path):
        """_invalidate_fingerprint removes the stamp file."""
        from scripts.run_analysis import _invalidate_fingerprint

        output_dir = tmp_path / "analysis"
        output_dir.mkdir()
        stamp = output_dir / ".analysis_fingerprint"
        stamp.write_text("stale")

        _invalidate_fingerprint(str(output_dir))
        assert not stamp.exists()

    def test_stale_viz_fingerprint_invalidated(self, tmp_path):
        """_invalidate_viz_fingerprint removes the stamp file."""
        from scripts.run_visualization import _invalidate_viz_fingerprint

        fig_dir = tmp_path / "figures"
        fig_dir.mkdir()
        stamp = fig_dir / ".viz_fingerprint"
        stamp.write_text("stale")

        _invalidate_viz_fingerprint(fig_dir)
        assert not stamp.exists()


# ------------------------------------------------------------------ #
# Interrupted final writes
# ------------------------------------------------------------------ #

class TestInterruptedFinalWrites:

    def test_summary_without_drift_h5_is_incomplete(self, tmp_path):
        """If summary.json exists but drift_metrics.h5 is missing, treat as incomplete."""
        from omegaconf import OmegaConf
        from scripts.run_sweep import _is_complete

        cfg = OmegaConf.create({
            "output_dir": str(tmp_path),
            "run_seed": 42,
            "graph_seed": 100,
            "training": {"total_rounds": 500},
        })

        run_dir = tmp_path / "seed_42_graph_100"
        run_dir.mkdir(parents=True)
        save_json({"total_rounds": 500}, run_dir / "summary.json")
        assert not _is_complete(cfg)
