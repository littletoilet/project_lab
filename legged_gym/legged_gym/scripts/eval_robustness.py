# SPDX-License-Identifier: BSD-3-Clause

import copy
import csv
import math
import os
import subprocess
import sys
from datetime import datetime

import isaacgym
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, get_load_path, task_registry


SCENARIOS = (
    "clean_rough",
    "hard_terrain",
    "low_friction",
    "payload_mass",
    "frequent_push",
    "mixed_stress",
)


def _set_eval_defaults(env_cfg):
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.complex_hard_terrain = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False


def _apply_hard_terrain(env_cfg):
    env_cfg.terrain.complex_hard_terrain = True
    env_cfg.terrain.terrain_proportions = [0.18, 0.27, 0.12, 0.08, 0.25, 0.10, 0.00]
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1


def _apply_low_friction(env_cfg):
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [0.2, 0.7]


def _apply_payload_mass(env_cfg):
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [1.0, 2.5]


def _apply_frequent_push(env_cfg):
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 5
    env_cfg.domain_rand.max_push_vel_xy = 1.5


def apply_scenario(env_cfg, scenario):
    _set_eval_defaults(env_cfg)
    if scenario == "clean_rough":
        return
    if scenario == "hard_terrain":
        _apply_hard_terrain(env_cfg)
        return
    if scenario == "low_friction":
        _apply_low_friction(env_cfg)
        return
    if scenario == "payload_mass":
        _apply_payload_mass(env_cfg)
        return
    if scenario == "frequent_push":
        _apply_frequent_push(env_cfg)
        return
    if scenario == "mixed_stress":
        _apply_hard_terrain(env_cfg)
        _apply_low_friction(env_cfg)
        _apply_payload_mass(env_cfg)
        _apply_frequent_push(env_cfg)
        return
    raise ValueError(f"Unknown scenario '{scenario}'. Expected one of: {', '.join(SCENARIOS)}")


def _load_runner(env, train_cfg, args):
    train_cfg.runner.resume = False
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    load_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    print(f"Loading model from: {load_path}")
    runner.load(load_path, map_location=env.device)
    policy = runner.get_inference_policy(device=env.device)
    return policy, train_cfg, load_path


def evaluate_once(args, scenario):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg = copy.deepcopy(env_cfg)
    train_cfg = copy.deepcopy(train_cfg)
    apply_scenario(env_cfg, scenario)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    policy, train_cfg, load_path = _load_runner(env, train_cfg, args)

    obs = env.get_observations()
    episode_returns = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    previous_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    completed_returns = []
    completed_lengths = []
    completed_successes = []
    completed_falls = []

    lin_vel_error_sum = 0.0
    yaw_vel_error_sum = 0.0
    orientation_sum = 0.0
    action_rate_sum = 0.0
    total_env_steps = 0

    max_steps = int(env.max_episode_length * (math.ceil(args.episodes / env.num_envs) + 3))

    with torch.inference_mode():
        for _ in range(max_steps):
            actions = policy(obs)
            action_rate = torch.sum(torch.square(actions - previous_actions), dim=1)
            obs, rewards, dones, infos = env.step(actions.detach())

            episode_returns += rewards
            episode_lengths += 1

            lin_vel_error_sum += torch.norm(env.commands[:, :2] - env.base_lin_vel[:, :2], dim=1).sum().item()
            yaw_vel_error_sum += torch.abs(env.commands[:, 2] - env.base_ang_vel[:, 2]).sum().item()
            orientation_sum += torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1).sum().item()
            action_rate_sum += action_rate.sum().item()
            total_env_steps += env.num_envs
            previous_actions[:] = actions[:]

            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                time_outs = infos.get("time_outs", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                for env_id in done_ids:
                    completed_returns.append(episode_returns[env_id].item())
                    completed_lengths.append(episode_lengths[env_id].item() * env.dt)
                    is_success = bool(time_outs[env_id].item())
                    completed_successes.append(float(is_success))
                    completed_falls.append(float(not is_success))

                episode_returns[done_ids] = 0.0
                episode_lengths[done_ids] = 0.0
                previous_actions[done_ids] = 0.0

                if len(completed_returns) >= args.episodes:
                    break

    if not completed_returns:
        raise RuntimeError("No completed episodes were collected. Increase --episodes or check the environment.")

    completed_returns = completed_returns[: args.episodes]
    completed_lengths = completed_lengths[: args.episodes]
    completed_successes = completed_successes[: args.episodes]
    completed_falls = completed_falls[: args.episodes]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": args.task,
        "scenario": scenario,
        "episodes": len(completed_returns),
        "num_envs": env.num_envs,
        "checkpoint": load_path,
        "mean_episode_reward": sum(completed_returns) / len(completed_returns),
        "mean_survival_time_s": sum(completed_lengths) / len(completed_lengths),
        "success_rate": sum(completed_successes) / len(completed_successes),
        "fall_rate": sum(completed_falls) / len(completed_falls),
        "mean_lin_vel_error": lin_vel_error_sum / total_env_steps,
        "mean_yaw_vel_error": yaw_vel_error_sum / total_env_steps,
        "mean_orientation_error": orientation_sum / total_env_steps,
        "mean_action_rate": action_rate_sum / total_env_steps,
    }
    return row


def _write_outputs(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "robustness_eval.csv")
    md_path = os.path.join(output_dir, "robustness_eval.md")

    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "a") as f:
        f.write("\n## Evaluation Batch\n\n")
        f.write("| task | scenario | episodes | reward | survival_s | success | fall | lin_err | yaw_err | orient_err | action_rate |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                "| {task} | {scenario} | {episodes} | {mean_episode_reward:.3f} | "
                "{mean_survival_time_s:.3f} | {success_rate:.3f} | {fall_rate:.3f} | "
                "{mean_lin_vel_error:.3f} | {mean_yaw_vel_error:.3f} | "
                "{mean_orientation_error:.3f} | {mean_action_rate:.3f} |\n".format(**row)
            )
        f.write("\n")
        for row in rows:
            f.write(f"- `{row['task']}` / `{row['scenario']}` checkpoint: `{row['checkpoint']}`\n")

    return csv_path, md_path


def _argv_for_scenario(scenario):
    argv = [sys.executable, os.path.abspath(__file__)]
    replaced = False
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--scenario":
            argv.extend(["--scenario", scenario])
            replaced = True
            i += 2
            continue
        if arg.startswith("--scenario="):
            argv.append(f"--scenario={scenario}")
            replaced = True
        else:
            argv.append(arg)
        i += 1

    if not replaced:
        argv.append(f"--scenario={scenario}")
    return argv


def _run_all_scenarios(args):
    output_dir = args.output_dir or os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "phase2_eval")
    print("Running all scenarios in separate Python processes to avoid Isaac Gym sim reuse crashes.", flush=True)
    for scenario in SCENARIOS:
        print(f"\n=== Evaluating scenario: {scenario} ===", flush=True)
        subprocess.run(_argv_for_scenario(scenario), check=True)

    print(f"\nWrote CSV: {os.path.join(output_dir, 'robustness_eval.csv')}", flush=True)
    print(f"Wrote Markdown: {os.path.join(output_dir, 'robustness_eval.md')}", flush=True)


def main():
    args = get_args()
    args.headless = True

    if args.scenario == "all":
        _run_all_scenarios(args)
        return

    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    output_dir = args.output_dir or os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "phase2_eval")
    rows = [evaluate_once(args, scenario) for scenario in scenarios]
    csv_path, md_path = _write_outputs(rows, output_dir)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()
