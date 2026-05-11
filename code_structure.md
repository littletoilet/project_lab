# 代码结构说明文档

主要工作相关代码都在/home/zzihan/data2/LAB_gongkechuang/legged_gym/legged_gym（见 [legged_gym/README.md](legged_gym/README.md)）。

## 1. env 结构

### 1.1 顶层入口
- 训练入口在 scripts/train.py，调用 task_registry.make_env() 创建环境，再调用 task_registry.make_alg_runner() 创建 PPO 训练器（见 [legged_gym/scripts/train.py](legged_gym/scripts/train.py)）。
- 运行/回放入口在 scripts/play.py，复用同样的 env 与训练配置（见 [legged_gym/scripts/play.py](legged_gym/scripts/play.py)）。

### 1.2 BaseTask
BaseTask 定义了所有环境的基础框架（见 [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)）。
- 初始化 Isaac Gym 仿真、设备、viewer（见 [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)）。
- 分配核心缓冲区：obs_buf、rew_buf、reset_buf、episode_length_buf、privileged_obs_buf 等（见 [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)）。
- 提供 reset()/render() 等基础接口（见 [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)）。
- get_observations() 默认返回 obs_buf（见 [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)）。

关键数据流：
- BaseTask 只负责通用仿真框架，不含具体机器人与奖励逻辑（见 [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)）。

### 1.3 LeggedRobot
LeggedRobot 是具体任务环境的基类，继承 BaseTask，主要负责（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- 解析 cfg（控制频率、奖励系数、指令范围等）（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- create_sim() 中创建 terrain 并创建所有 env/actor（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- step() -> post_physics_step() -> compute_reward()/compute_observations()，完成一轮交互（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。

核心流程简化（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）：
1) step(actions)（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
2) render() + physics simulate（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
3) post_physics_step()（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）：
   - refresh state tensors（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
   - 终止判定 check_termination()（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
   - 奖励 compute_reward()（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
   - reset 需要 reset 的 env（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
   - 观测 compute_observations()（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
4) 返回 obs/reward/done/extras（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。

### 1.4 配置体系
配置采用类嵌套结构（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- LeggedRobotCfg 定义通用默认值（obs 维度、地形参数、奖励系数、噪声等）（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- AnymalCRoughCfg 等任务在子类中覆写关键字段（asset 路径、terrain 类型等）（见 [legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py](legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)）。
- LeggedRobotCfgPPO 定义算法和 runner 参数（batch、lr、迭代次数等）（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。

## 2. reward 计算

### 2.1 奖励权重来源
奖励项的权重来自 cfg.rewards.scales，默认在 LeggedRobotCfg.rewards.scales 中定义，典型包含（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- tracking_lin_vel / tracking_ang_vel（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- lin_vel_z / ang_vel_xy（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- torques / dof_vel / dof_acc（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- collision / feet_air_time / action_rate（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- termination（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。

### 2.2 奖励准备阶段
初始化时调用 _prepare_reward_function()（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- 移除 scale=0 的项（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- 其余 scale 乘以 dt（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- 收集所有 reward function（按命名规则 _reward_<name>）（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- 初始化 episode_sums 用于统计（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。

### 2.3 单步计算逻辑
compute_reward() 内部逻辑（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- 对每个 reward 函数计算 rew = func() * scale（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- 累积到 rew_buf 与 episode_sums（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- 可选裁剪 only_positive_rewards（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- 若存在 termination，最后叠加终止奖励（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。

### 2.4 reward 函数实现方式
奖励函数集中定义在 LeggedRobot 中，命名统一（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- _reward_tracking_lin_vel() / _reward_tracking_ang_vel()（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- _reward_action_rate() / _reward_torques()（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- _reward_collision() / _reward_feet_air_time() 等（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。

这些函数通常以当前状态张量（速度、姿态、力、足端接触等）计算即时 reward（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。

## 3. terrain 生成逻辑

### 3.1 Terrain 类入口
Terrain 位于 legged_gym/utils/terrain.py（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
初始化参数来自 cfg.terrain，决定（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py) 和 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- 地形类型 mesh_type (plane/heightfield/trimesh)（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- 地形分辨率 (horizontal_scale/vertical_scale)（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- 网格布局 (num_rows/num_cols)（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- curriculum / selected / randomized 模式（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。

### 3.2 地形生成模式
Terrain.__init__() 中根据配置分支（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
- curriculum=True: 调用 curiculum()，按行递增难度（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
- selected=True: 调用 selected_terrain()，固定地形类型（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
- 默认: randomized_terrain()，按比例随机（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。

### 3.3 地形类型构成
make_terrain() 通过 choice + difficulty 生成单块子地形（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
- slope / rough slope（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
- pyramid stairs（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
- discrete obstacles（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
- stepping stones（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
- gap / pit（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
这些子地形使用 isaacgym.terrain_utils 构造，再写入总 height_field_raw（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。

### 3.4 映射与原点
add_terrain_to_map()（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
- 将单个 sub-terrain 填充到总 height field（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
- 计算 env_origin_z 作为机器人出生高度（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。

### 3.5 在仿真中使用
LeggedRobot.create_sim() 中（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- mesh_type == heightfield: add_heightfield（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- mesh_type == trimesh: add_triangle_mesh（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- mesh_type == plane: add_ground（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。

不同地形决定观测中是否包含高度测量点（measure_heights）（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) 和 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。

## 4. 推荐阅读路径
若需要进一步深入，可按以下顺序阅读源码（见 [legged_gym/README.md](legged_gym/README.md)）。
1) envs/base/base_task.py（见 [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)）。
2) envs/base/legged_robot.py（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
3) envs/base/legged_robot_config.py（见 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
4) utils/terrain.py（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。
5) envs/anymal_c/mixed_terrains/anymal_c_rough_config.py（见 [legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py](legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)）。

## 5. 总结
- env 结构采用 BaseTask + LeggedRobot 的两层抽象（见 [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py) 和 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）。
- reward 由 cfg 驱动，函数名与 scale 键严格对应（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) 和 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。
- terrain 生成支持随机/课程/固定三种模式，可直接控制难度分布（见 [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)）。

如需补充更细的 reward 公式推导、观测维度拆解或 terrain 参数调优建议，可继续指定（见 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) 和 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)）。