"""No optimizer or simulator is run by these tests."""
import importlib.util
import json
from pathlib import Path


def launcher():
    spec = importlib.util.spec_from_file_location("pronk_imitation_launcher", Path(__file__).with_name("train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nominal_train_keeps_all_200_anchors_and_completes_warmup():
    module = launcher()
    cfg = json.loads((module.ROOT / module.P["base_config"]).read_text(encoding="utf-8"))
    result = module.budget(cfg, module.P["episodes"])
    assert result["unique_anchors"] == 669000
    assert result["samples_after_symmetry"] == 2676000
    assert result["train_episodes"] == 3178
    assert result["validation_episodes"] == 167
    assert result["steps_per_epoch"] == 621
    assert result["total_updates"] == 31050
    assert result["lr_warmup_updates"] == 10000


def test_only_one_frozen_dataset_no_resume_or_recovery_flags():
    module = launcher()
    command = module.training_command(Path("unused_test_output"), "cuda:0", True)
    assert command[command.index("--datasets") + 1].endswith("pronk_full.hdf5")
    assert command[command.index("--policy_backend") + 1] == "diffusion"
    assert command[command.index("--seed") + 1] == "42"
    assert not any("recovery" in x or "checkpoint" in x or "resume" in x for x in command)
