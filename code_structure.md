# Code Structure Documentation

The main work-related code is located in `/home/zzihan/data2/LAB_gongkechuang/legged_gym/legged_gym` (see [legged_gym/README.md](legged_gym/README.md)).

## 1. Environment Structure

### 1.1 Top-Level Entry Points
- The training entry point is `scripts/train.py`. It calls `task_registry.make_env()` to create the environment and then calls `task_registry.make_alg_runner()` to create the PPO trainer (see [legged_gym/scripts/train.py](legged_gym/scripts/train.py)).
- The execution/replay entry point is `scripts/play.py`, which reuses the same environment and training configuration (see [legged_gym/scripts/play.py](legged_gym/scripts/play.py)).

### 1.2 BaseTask
`BaseTask` defines the basic framework for all environments (see [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)).
- Initializes the Isaac Gym simulation, device, and viewer (see [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)).
- Allocates core buffers such as `obs_buf`, `rew_buf`, `reset_buf`, `episode_length_buf`, and `privileged_obs_buf` (see [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)).
- Provides basic interfaces such as `reset()` and `render()` (see [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)).
- `get_observations()` returns `obs_buf` by default (see [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)).

Key data flow:
- `BaseTask` is only responsible for the general simulation framework. It does not contain robot-specific or reward-specific logic (see [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)).

### 1.3 LeggedRobot
`LeggedRobot` is the base class for concrete task environments. It inherits from `BaseTask` and is mainly responsible for the following (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- Parsing `cfg`, including control frequency, reward coefficients, command ranges, and related settings (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- Creating the terrain and all envs/actors in `create_sim()` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- Completing one interaction step through `step() -> post_physics_step() -> compute_reward()/compute_observations()` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).

Simplified core workflow (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)):
1) `step(actions)` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
2) `render()` + physics simulation (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
3) `post_physics_step()` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)):
   - Refresh state tensors (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
   - Perform termination checking via `check_termination()` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
   - Compute rewards via `compute_reward()` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
   - Reset the environments that need to be reset (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
   - Compute observations via `compute_observations()` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
4) Return `obs`, `reward`, `done`, and `extras` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).

### 1.4 Configuration System
The configuration system uses nested classes (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- `LeggedRobotCfg` defines general default values, including observation dimensions, terrain parameters, reward coefficients, and noise settings (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- Task-specific configurations such as `AnymalCRoughCfg` override key fields in subclasses, such as the asset path and terrain type (see [legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py](legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)).
- `LeggedRobotCfgPPO` defines algorithm and runner parameters, including batch size, learning rate, and number of iterations (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).

## 2. Reward Computation

### 2.1 Source of Reward Weights
The weights of reward terms come from `cfg.rewards.scales`. By default, they are defined in `LeggedRobotCfg.rewards.scales` and typically include the following (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- `tracking_lin_vel` / `tracking_ang_vel` (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- `lin_vel_z` / `ang_vel_xy` (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- `torques` / `dof_vel` / `dof_acc` (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- `collision` / `feet_air_time` / `action_rate` (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- `termination` (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).

### 2.2 Reward Preparation Stage
During initialization, `_prepare_reward_function()` is called (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- Remove items whose `scale=0` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- Multiply the remaining scales by `dt` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- Collect all reward functions according to the naming convention `_reward_<name>` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- Initialize `episode_sums` for statistics (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).

### 2.3 Single-Step Computation Logic
The internal logic of `compute_reward()` is as follows (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- For each reward function, compute `rew = func() * scale` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- Accumulate the result into `rew_buf` and `episode_sums` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- Optionally clip rewards using `only_positive_rewards` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- If `termination` exists, add the termination reward at the end (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).

### 2.4 Reward Function Implementation
Reward functions are centrally defined in `LeggedRobot` and follow a unified naming convention (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- `_reward_tracking_lin_vel()` / `_reward_tracking_ang_vel()` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- `_reward_action_rate()` / `_reward_torques()` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- `_reward_collision()` / `_reward_feet_air_time()`, etc. (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).

These functions usually compute the instantaneous reward based on current state tensors, such as velocity, posture, force, and foot contact information (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).

## 3. Terrain Generation Logic

### 3.1 Terrain Class Entry Point
`Terrain` is located in `legged_gym/utils/terrain.py` (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
Its initialization parameters come from `cfg.terrain` and determine the following (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py) and [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- Terrain type `mesh_type` (`plane`/`heightfield`/`trimesh`) (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- Terrain resolution (`horizontal_scale`/`vertical_scale`) (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- Grid layout (`num_rows`/`num_cols`) (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- `curriculum` / `selected` / `randomized` modes (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).

### 3.2 Terrain Generation Modes
In `Terrain.__init__()`, the generation branch is selected according to the configuration (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
- `curriculum=True`: calls `curiculum()` and increases difficulty row by row (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
- `selected=True`: calls `selected_terrain()` and uses a fixed terrain type (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
- Default: calls `randomized_terrain()` and samples terrains randomly according to predefined proportions (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).

### 3.3 Terrain Type Composition
`make_terrain()` generates a single sub-terrain based on `choice` and `difficulty` (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
- Slope / rough slope (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
- Pyramid stairs (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
- Discrete obstacles (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
- Stepping stones (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
- Gap / pit (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
These sub-terrains are constructed using `isaacgym.terrain_utils` and then written into the overall `height_field_raw` (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).

### 3.4 Mapping and Origins
`add_terrain_to_map()` (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
- Fills a single sub-terrain into the overall height field (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
- Computes `env_origin_z` as the robot spawn height (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).

### 3.5 Usage in Simulation
In `LeggedRobot.create_sim()` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)):
- `mesh_type == heightfield`: use `add_heightfield` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- `mesh_type == trimesh`: use `add_triangle_mesh` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- `mesh_type == plane`: use `add_ground` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).

Different terrains determine whether height measurement points, i.e., `measure_heights`, are included in the observations (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) and [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).

## 4. Recommended Reading Path
For deeper understanding, read the source code in the following order (see [legged_gym/README.md](legged_gym/README.md)).
1) `envs/base/base_task.py` (see [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)).
2) `envs/base/legged_robot.py` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
3) `envs/base/legged_robot_config.py` (see [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
4) `utils/terrain.py` (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).
5) `envs/anymal_c/mixed_terrains/anymal_c_rough_config.py` (see [legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py](legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)).

## 5. Summary
- The environment structure uses a two-level abstraction: `BaseTask + LeggedRobot` (see [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py) and [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)).
- Rewards are driven by `cfg`, and reward function names strictly correspond to the keys in `scales` (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) and [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
- Terrain generation supports three modes: randomized, curriculum-based, and fixed/selected. These modes can directly control the difficulty distribution (see [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)).

For additional details such as reward formula derivations, observation dimension breakdowns, or terrain parameter tuning suggestions, please specify the desired part (see [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) and [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)).
