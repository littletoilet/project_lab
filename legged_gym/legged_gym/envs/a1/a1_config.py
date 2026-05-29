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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            orientation = -0.2

class A1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_a1'


class A1TerrainCfg( A1RoughCfg ):
    class terrain( A1RoughCfg.terrain ):
        terrain_proportions = [0.05, 0.10, 0.25, 0.20, 0.20, 0.10, 0.10]
        max_init_terrain_level = 7


class A1FrictionCfg( A1RoughCfg ):
    class domain_rand( A1RoughCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.3, 1.5]
        randomize_base_mass = False
        push_robots = True


class A1MassCfg( A1RoughCfg ):
    class domain_rand( A1RoughCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.5]
        push_robots = True


class A1PushCfg( A1RoughCfg ):
    class domain_rand( A1RoughCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 1.5


class A1RewardSmoothCfg( A1RoughCfg ):
    class rewards( A1RoughCfg.rewards ):
        class scales( A1RoughCfg.rewards.scales ):
            orientation = -0.4
            ang_vel_xy = -0.10
            action_rate = -0.03
            dof_acc = -5e-7


class A1RobustAllCfg( A1RoughCfg ):
    class terrain( A1RoughCfg.terrain ):
        terrain_proportions = [0.05, 0.10, 0.25, 0.20, 0.20, 0.10, 0.10]
        max_init_terrain_level = 7

    class domain_rand( A1RoughCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.3, 1.5]
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.5]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 1.5

    class rewards( A1RoughCfg.rewards ):
        class scales( A1RoughCfg.rewards.scales ):
            orientation = -0.4
            ang_vel_xy = -0.10
            action_rate = -0.03
            dof_acc = -5e-7


class A1TerrainCfgPPO( A1RoughCfgPPO ):
    class runner( A1RoughCfgPPO.runner ):
        run_name = 'phase2_terrain'
        experiment_name = 'ablation_a1_terrain'
        max_iterations = 500


class A1FrictionCfgPPO( A1RoughCfgPPO ):
    class runner( A1RoughCfgPPO.runner ):
        run_name = 'phase2_friction'
        experiment_name = 'ablation_a1_friction'
        max_iterations = 500


class A1MassCfgPPO( A1RoughCfgPPO ):
    class runner( A1RoughCfgPPO.runner ):
        run_name = 'phase2_mass'
        experiment_name = 'ablation_a1_mass'
        max_iterations = 500


class A1PushCfgPPO( A1RoughCfgPPO ):
    class runner( A1RoughCfgPPO.runner ):
        run_name = 'phase2_push'
        experiment_name = 'ablation_a1_push'
        max_iterations = 500


class A1RewardSmoothCfgPPO( A1RoughCfgPPO ):
    class runner( A1RoughCfgPPO.runner ):
        run_name = 'phase2_reward_smooth'
        experiment_name = 'ablation_a1_reward_smooth'
        max_iterations = 500


class A1RobustAllCfgPPO( A1RoughCfgPPO ):
    class runner( A1RoughCfgPPO.runner ):
        run_name = 'phase2_all'
        experiment_name = 'robust_a1_all'
        max_iterations = 500

  
