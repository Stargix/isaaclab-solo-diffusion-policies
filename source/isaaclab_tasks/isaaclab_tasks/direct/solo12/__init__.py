# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="solo12-v0",
    entry_point=f"{__name__}.solo12_env:Solo12Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo12_env_cfg:Solo12EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Solo12PPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Solo12PPORunnerWithSymmetryCfg"
        ),
    },
)


def _register(task_id: str, cfg_name: str, rsl_rl_cfg_name: str):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.solo12_env:Solo12Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.solo12_env_cfg:{cfg_name}",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{rsl_rl_cfg_name}",
        },
    )


_register("solo12-IMU-based-teacher", "Solo12BaseImuTeacherEnvCfg", "Solo12BaseImuTeacherPPORunnerCfg")
_register("solo12-IMU-student-rl", "Solo12BaseImuStudentRlEnvCfg", "Solo12BaseImuStudentRlPPORunnerCfg")
_register("solo12-IMU-student-dagger", "Solo12BaseImuStudentDaggerEnvCfg", "Solo12BaseImuTeacherPPORunnerCfg")
_register("Isaac-Solo12-BaseIMU-Teacher-Direct-v0", "Solo12BaseImuTeacherEnvCfg", "Solo12BaseImuTeacherPPORunnerCfg")
_register("Isaac-Solo12-BaseIMU-StudentRL-Direct-v0", "Solo12BaseImuStudentRlEnvCfg", "Solo12BaseImuStudentRlPPORunnerCfg")

gym.register(
    id="solo12-crouch-v0",
    entry_point=f"{__name__}.solo12_crouch_env:Solo12CrouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo12_crouch_env:Solo12CrouchEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_fast.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.solo12_crouch_env:Solo12CrouchPPORunnerCfg",
    },
)

gym.register(
    id="solo12-sprint-v0",
    entry_point=f"{__name__}.solo12_sprint_env:Solo12SprintEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.solo12_sprint_env:Solo12SprintEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_fast.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.solo12_sprint_env:Solo12SprintPPORunnerCfg",
    },
)



