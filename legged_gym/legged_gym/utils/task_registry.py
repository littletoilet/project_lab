# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from datetime import datetime
from typing import Optional, Tuple
import torch
import numpy as np
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class _RslRlEnvWrapper(VecEnv):
    """Adapt legacy legged_gym envs to the rsl_rl VecEnv TensorDict API."""

    def __init__(self, env: VecEnv):
        self._env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.device = env.device
        self.cfg = env.cfg

    def get_observations(self) -> TensorDict:
        obs = self._env.get_observations()
        privileged = None
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, privileged = obs
        return self._as_tensordict(obs, privileged)

    def step(self, actions: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        result = self._env.step(actions)
        if isinstance(result, tuple) and len(result) == 5:
            obs, privileged, rewards, dones, extras = result
        else:
            obs, rewards, dones, extras = result
            privileged = None
        obs_td = self._as_tensordict(obs, privileged)
        return obs_td, rewards, dones, extras

    def _as_tensordict(self, obs: torch.Tensor, privileged: Optional[torch.Tensor]) -> TensorDict:
        if isinstance(obs, TensorDict):
            if privileged is not None and "privileged" not in obs.keys():
                obs = obs.clone()
                obs.set("privileged", privileged)
            return obs
        data = {"policy": obs}
        if privileged is not None:
            data["privileged"] = privileged
        return TensorDict(data, batch_size=[self.num_envs], device=obs.device)

    def __getattr__(self, name: str):
        return getattr(self._env, name)

def _upgrade_train_cfg(train_cfg_dict: dict) -> dict:
    """Upgrade legacy legged_gym PPO config to the current rsl_rl format."""
    if "algorithm" not in train_cfg_dict:
        return train_cfg_dict

    runner_cfg = train_cfg_dict.get("runner", {})
    policy_cfg = train_cfg_dict.get("policy", {})
    alg_cfg = dict(train_cfg_dict.get("algorithm", {}))

    is_legacy = "runner" in train_cfg_dict and "actor" not in train_cfg_dict and "critic" not in train_cfg_dict

    if is_legacy:
        alg_cfg.setdefault("class_name", runner_cfg.get("algorithm_class_name", "PPO"))
        alg_cfg.setdefault("rnd_cfg", None)
        alg_cfg.setdefault("symmetry_cfg", None)

        policy_class_name = runner_cfg.get("policy_class_name", "ActorCritic")
        if policy_class_name in ("MLPModel", "RNNModel", "CNNModel"):
            model_class = policy_class_name
        elif policy_class_name in ("ActorCriticRecurrent", "RecurrentActorCritic"):
            model_class = "RNNModel"
        else:
            model_class = "MLPModel"

        actor_cfg = {
            "class_name": model_class,
            "hidden_dims": policy_cfg.get("actor_hidden_dims", policy_cfg.get("hidden_dims", [256, 256, 256])),
            "activation": policy_cfg.get("activation", "elu"),
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": policy_cfg.get("init_noise_std", 1.0),
            },
        }
        critic_cfg = {
            "class_name": model_class,
            "hidden_dims": policy_cfg.get("critic_hidden_dims", policy_cfg.get("hidden_dims", [256, 256, 256])),
            "activation": policy_cfg.get("activation", "elu"),
        }

        if model_class == "RNNModel":
            if "rnn_type" in policy_cfg:
                actor_cfg["rnn_type"] = policy_cfg.get("rnn_type")
                critic_cfg["rnn_type"] = policy_cfg.get("rnn_type")
            if "rnn_hidden_size" in policy_cfg:
                actor_cfg["rnn_hidden_dim"] = policy_cfg.get("rnn_hidden_size")
                critic_cfg["rnn_hidden_dim"] = policy_cfg.get("rnn_hidden_size")
            if "rnn_num_layers" in policy_cfg:
                actor_cfg["rnn_num_layers"] = policy_cfg.get("rnn_num_layers")
                critic_cfg["rnn_num_layers"] = policy_cfg.get("rnn_num_layers")

        train_cfg_dict["algorithm"] = alg_cfg
        train_cfg_dict["actor"] = actor_cfg
        train_cfg_dict["critic"] = critic_cfg
        train_cfg_dict["obs_groups"] = train_cfg_dict.get("obs_groups", {"actor": ["policy"], "critic": ["policy"]})
        train_cfg_dict["num_steps_per_env"] = runner_cfg.get("num_steps_per_env")
        train_cfg_dict["save_interval"] = runner_cfg.get("save_interval", 50)
    else:
        alg_cfg.setdefault("class_name", "PPO")
        alg_cfg.setdefault("rnd_cfg", None)
        alg_cfg.setdefault("symmetry_cfg", None)
        train_cfg_dict["algorithm"] = alg_cfg
        train_cfg_dict.setdefault("obs_groups", {"actor": ["policy"], "critic": ["policy"]})
        train_cfg_dict.setdefault("num_steps_per_env", runner_cfg.get("num_steps_per_env", 24))
        train_cfg_dict.setdefault("save_interval", runner_cfg.get("save_interval", 50))

    return train_cfg_dict


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                    sim_params=sim_params,
                    physics_engine=args.physics_engine,
                    sim_device=args.sim_device,
                    headless=args.headless)
        env = _RslRlEnvWrapper(env)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        train_cfg_dict = _upgrade_train_cfg(class_to_dict(train_cfg))
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()