"""Tests for checkpoint resume, sweep skip, and idempotent analysis.

All tests run on CPU with synthetic data, no downloads required.
"""

from __future__ import annotations

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
    import copy

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    clients = []
    for cid in range(n):
        # Each client gets a deep copy of the model and its own small dataset.
        m = copy.deepcopy(model)
        data = torch.randn(32, 8)
        targets = torch.randint(0, 4, (32,))
        ds = TensorDataset(data, targets)
        ds.targets = targets.tolist()
        ds.run_seed = seed
        ds.set_round = lambda r: None  # no-op

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


# ------------------------------------------------------------------ #
# TrainingCheckpointManager unit tests
# ------------------------------------------------------------------ #

class TestTrainingCheckpointManager:

    def test_save_and_load(self, tmp_output):
        cfg = {"lr": 0.01, "rounds": 10}
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict=cfg)
        clients = _make_clients(n=2)

        # Run one training step so optimizers have state.
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

    def test_corrupt_checkpoint_skipped(self, tmp_output):
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict={"a": 1})
        # Write a corrupt file.
        p = mgr.output_dir / "resume_ckpt_r0010.pt"
        p.write_text("not a valid checkpoint")

        assert mgr.find_latest() is None

    def test_restore_clients(self, tmp_output):
        cfg = {"x": 1}
        mgr = TrainingCheckpointManager(str(tmp_output), config_dict=cfg)
        clients = _make_clients(n=2)
        for c in clients:
            c.local_step(0)

        # Save state.
        mgr.save_training_state(clients, round_num=3, history={})

        # Modify client weights to something different.
        for c in clients:
            with torch.no_grad():
                for p in c.model.parameters():
                    p.fill_(999.0)

        # Restore.
        ckpt = mgr.load_training_state()
        mgr.restore_clients(clients, ckpt)

        # Weights should no longer be 999.
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

        # Only the last 2 should remain.
        remaining = sorted(mgr.output_dir.glob("resume_ckpt_r*.pt"))
        assert len(remaining) == 2
        assert "r0020" in remaining[0].name
        assert "r0030" in remaining[1].name


# ------------------------------------------------------------------ #
# DSGDCoordinator resume tests
# ------------------------------------------------------------------ #

class TestCoordinatorResume:

    def _make_coordinator(self, tmp_output, total_rounds=20, checkpoint_every=5):
        n = 4
        clients = _make_clients(n=n)
        W = np.ones((n, n), dtype=np.float64) / n
        lr_schedule = CosineDecaySchedule(lr_max=0.01, lr_min=0.001, total_rounds=total_rounds)

        resume_mgr = TrainingCheckpointManager(
            str(tmp_output / "resume_checkpoints"),
            config_dict={"total_rounds": total_rounds},
        )

        coordinator = DSGDCoordinator(
            clients=clients,
            mixing_matrix=W,
            lr_schedule=lr_schedule,
            total_rounds=total_rounds,
            eval_every=100,   # no evals to keep test fast
            checkpoint_every=checkpoint_every,
            output_dir=str(tmp_output),
            resume_manager=resume_mgr,
        )
        return coordinator, resume_mgr

    def test_full_run_creates_checkpoint(self, tmp_output):
        coord, resume_mgr = self._make_coordinator(tmp_output)
        history = coord.run()

        assert len(history["round"]) == 20
        assert resume_mgr.find_latest() is not None

    def test_resume_skips_completed_rounds(self, tmp_output):
        """Run 10 rounds, checkpoint, then create a new coordinator that
        resumes and only runs rounds 10–19."""
        coord1, _ = self._make_coordinator(tmp_output, total_rounds=10, checkpoint_every=5)
        h1 = coord1.run()
        assert len(h1["round"]) == 10

        # Create a new coordinator pointing at the same checkpoint dir
        # but with total_rounds=20. It should resume from round 10.
        n = 4
        clients = _make_clients(n=n)
        W = np.ones((n, n), dtype=np.float64) / n
        lr_schedule = CosineDecaySchedule(lr_max=0.01, lr_min=0.001, total_rounds=20)
        resume_mgr2 = TrainingCheckpointManager(
            str(tmp_output / "resume_checkpoints"),
            config_dict={"total_rounds": 10},  # same config hash as original
        )
        coord2 = DSGDCoordinator(
            clients=clients, mixing_matrix=W, lr_schedule=lr_schedule,
            total_rounds=20, eval_every=100, checkpoint_every=5,
            output_dir=str(tmp_output), resume_manager=resume_mgr2,
        )
        h2 = coord2.run()

        # History should contain rounds 10–19 from the new coordinator
        # (history from checkpoint + new rounds).
        assert h2["round"][-1] == 19
        assert h2["round"][0] == 0  # includes restored history

    def test_already_complete_run_is_noop(self, tmp_output):
        """If the checkpoint shows training is already done, run() returns immediately."""
        coord, _ = self._make_coordinator(tmp_output, total_rounds=10)
        h1 = coord.run()
        assert len(h1["round"]) == 10

        # New coordinator with same total_rounds should detect completion.
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
        # Should return the restored history without running any new rounds.
        assert h2["round"][-1] == 9


# ------------------------------------------------------------------ #
# Sweep-level skip/resume tests
# ------------------------------------------------------------------ #

class TestSweepResume:

    def test_is_complete_requires_valid_json(self, tmp_path):
        """_is_complete returns False for missing, corrupt, or incompatible summary.json."""
        from omegaconf import OmegaConf

        # Build a minimal OmegaConf config.
        cfg = OmegaConf.create({
            "output_dir": str(tmp_path),
            "run_seed": 42,
            "graph_seed": 100,
            "training": {"total_rounds": 500},
        })

        from scripts.run_sweep import _is_complete

        # No summary at all.
        assert not _is_complete(cfg)

        # Create the run directory and a corrupt summary.
        run_dir = tmp_path / "seed_42_graph_100"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text("NOT JSON")
        assert not _is_complete(cfg)

        # Valid JSON but wrong total_rounds.
        save_json({"total_rounds": 100}, run_dir / "summary.json")
        assert not _is_complete(cfg)

        # Matching total_rounds.
        save_json({"total_rounds": 500}, run_dir / "summary.json")
        assert _is_complete(cfg)

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

        # Create a resume checkpoint.
        ckpt_dir = tmp_path / "seed_42_graph_100" / "resume_checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "resume_ckpt_r0050.pt").write_bytes(b"dummy")
        assert _has_resume_checkpoint(cfg)
        assert _run_status(cfg) == "interrupted"


# ------------------------------------------------------------------ #
# Atomic save_json test
# ------------------------------------------------------------------ #

class TestAtomicSaveJson:

    def test_no_partial_file_on_crash(self, tmp_path):
        """save_json writes atomically via a temp file."""
        target = tmp_path / "test.json"
        save_json({"a": 1}, target)
        assert load_json(target) == {"a": 1}

        # The .tmp file should not linger.
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
