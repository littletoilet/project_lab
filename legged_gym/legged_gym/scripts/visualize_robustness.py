# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import subprocess
import sys
from datetime import datetime

import isaacgym
from isaacgym import gymapi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.scripts.eval_robustness import SCENARIOS, _load_runner, apply_scenario
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.terrain import Terrain


ABLATION_TASKS = (
    ("terrain", "hard_terrain", "a1_terrain"),
    ("friction", "low_friction", "a1_friction"),
    ("mass", "payload_mass", "a1_mass"),
    ("push", "frequent_push", "a1_push"),
    ("smooth", "clean_rough", "a1_reward_smooth"),
)

SCENARIO_NOTES = {
    "clean_rough": "Nominal rough terrain; no extra randomization or pushes.",
    "hard_terrain": "Blocky rough terrain with denser stairs, slopes, obstacles, shallow pits, and extra rectangular bumps.",
    "low_friction": "Rough terrain with friction randomized to 0.2-0.7.",
    "payload_mass": "Rough terrain with +1.0 to +2.5 kg base payload.",
    "frequent_push": "Rough terrain with 1.5 m/s pushes every 5 seconds.",
    "mixed_stress": "Hard terrain, low friction, payload, and frequent pushes together.",
}


def _output_dir(args):
    return args.output_dir or os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "phase2_visualization")


def _sanitize(name):
    return name.replace("/", "_").replace(" ", "_")


def _set_viz_defaults(env_cfg, args):
    env_cfg.env.num_envs = args.num_envs or 25
    env_cfg.viewer.enable_camera_sensors = True
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False


def _terrain_height_m(terrain):
    return terrain.height_field_raw.astype(np.float32) * terrain.cfg.vertical_scale


def write_terrain_overview(args, scenario):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = copy.deepcopy(env_cfg)
    apply_scenario(env_cfg, scenario)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False

    if args.seed is not None:
        np.random.seed(args.seed)

    terrain = Terrain(env_cfg.terrain, env_cfg.terrain.num_rows * env_cfg.terrain.num_cols)
    heights = _terrain_height_m(terrain)
    out_dir = os.path.join(_output_dir(args), "terrain_overviews")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scenario}.png")

    fig = plt.figure(figsize=(11.5, 7.0), dpi=140)
    ax = fig.add_axes([0.06, 0.09, 0.66, 0.82])
    im = ax.imshow(heights, cmap="terrain", origin="lower")
    ax.set_title(f"{scenario}: terrain layout", fontsize=13)
    ax.set_xlabel("terrain y cells")
    ax.set_ylabel("terrain x cells")
    cell_x = terrain.length_per_env_pixels
    cell_y = terrain.width_per_env_pixels
    border = terrain.border
    for row in range(env_cfg.terrain.num_rows + 1):
        ax.axhline(border + row * cell_x, color="white", linewidth=0.45, alpha=0.7)
    for col in range(env_cfg.terrain.num_cols + 1):
        ax.axvline(border + col * cell_y, color="white", linewidth=0.45, alpha=0.7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.025)
    cbar.set_label("height (m)")

    note_ax = fig.add_axes([0.77, 0.13, 0.2, 0.75])
    note_ax.axis("off")
    note_ax.text(0.0, 0.96, "Scenario", fontsize=12, weight="bold")
    note_ax.text(0.0, 0.86, SCENARIO_NOTES[scenario], fontsize=10, wrap=True)
    note_ax.text(0.0, 0.62, "Terrain mix", fontsize=12, weight="bold")
    if getattr(env_cfg.terrain, "complex_hard_terrain", False):
        terrain_labels = "slope / stairs / obstacles / stones / shallow pits / extra blocks"
        terrain_mix = "standard block terrains plus extra rectangular bumps and shallow cutouts"
    else:
        terrain_labels = "slope / rough slope / stairs / obstacles / stones / gaps / pits"
        terrain_mix = str(env_cfg.terrain.terrain_proportions)
    note_ax.text(0.0, 0.52, terrain_labels, fontsize=9, wrap=True)
    note_ax.text(0.0, 0.34, terrain_mix, fontsize=9, wrap=True)
    note_ax.text(0.0, 0.16, "Grid cells show the test sub-terrains used for rollouts.", fontsize=9, wrap=True)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _create_camera(env, args):
    camera_env_id = min(max(args.camera_env, 0), env.num_envs - 1)
    camera_props = gymapi.CameraProperties()
    camera_props.width = args.frame_width
    camera_props.height = args.frame_height
    camera_props.horizontal_fov = 75.0
    camera_env = env.envs[camera_env_id]
    camera_handle = env.gym.create_camera_sensor(camera_env, camera_props)
    return camera_env_id, camera_env, camera_handle


def _destroy_gym_resources(env, camera_env=None, camera_handle=None):
    if env is None:
        return
    if camera_handle is not None:
        try:
            env.gym.destroy_camera_sensor(env.sim, camera_env, camera_handle)
        except Exception as exc:
            print(f"Warning: failed to destroy camera sensor cleanly: {exc}", flush=True)
    viewer = getattr(env, "viewer", None)
    if viewer is not None:
        try:
            env.gym.destroy_viewer(viewer)
        except Exception as exc:
            print(f"Warning: failed to destroy viewer cleanly: {exc}", flush=True)
    sim = getattr(env, "sim", None)
    if sim is not None:
        try:
            env.gym.destroy_sim(sim)
        except Exception as exc:
            print(f"Warning: failed to destroy sim cleanly: {exc}", flush=True)


def _exit_successfully_without_atexit():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _set_follow_camera(env, camera_env_id, camera_env, camera_handle):
    robot_pos = env.root_states[camera_env_id, :3].detach().cpu().numpy()
    target = robot_pos + np.array([0.35, 0.0, 0.35], dtype=np.float32)
    camera_pos = target + np.array([-2.4, -2.0, 1.15], dtype=np.float32)
    env.gym.set_camera_location(
        camera_handle,
        camera_env,
        gymapi.Vec3(*camera_pos),
        gymapi.Vec3(*target),
    )


def _plot_rollout_traces(trace, out_path, task, scenario):
    if not trace["time"]:
        return
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.2), dpi=130, sharex=True)
    axes[0].plot(trace["time"], trace["lin_vel_error"], color="#2563eb", linewidth=1.4)
    axes[0].set_ylabel("lin vel err")
    axes[1].plot(trace["time"], trace["orientation_error"], color="#16a34a", linewidth=1.4)
    axes[1].set_ylabel("orientation err")
    axes[2].plot(trace["time"], trace["action_rate"], color="#dc2626", linewidth=1.4)
    axes[2].set_ylabel("action rate")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{task} on {scenario}: rollout diagnostics", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def record_rollout_frames(args, exit_on_success=False):
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg = copy.deepcopy(env_cfg)
    train_cfg = copy.deepcopy(train_cfg)
    apply_scenario(env_cfg, args.scenario)
    _set_viz_defaults(env_cfg, args)

    item_name = f"{_sanitize(args.scenario)}__{_sanitize(args.task)}"
    frames_dir = os.path.join(_output_dir(args), "frames", item_name)
    os.makedirs(frames_dir, exist_ok=True)
    for frame_name in os.listdir(frames_dir):
        if frame_name.endswith(".png"):
            os.remove(os.path.join(frames_dir, frame_name))

    env = None
    camera_env = None
    camera_handle = None
    try:
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        policy, train_cfg, load_path = _load_runner(env, train_cfg, args)
        obs = env.get_observations()

        camera_env_id, camera_env, camera_handle = _create_camera(env, args)
        previous_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        trace = {"time": [], "lin_vel_error": [], "orientation_error": [], "action_rate": []}
        saved_frames = 0
        max_steps = max(1, args.max_frames) * max(1, args.frame_stride)

        with torch.inference_mode():
            for step in range(max_steps):
                actions = policy(obs.detach())
                action_rate = torch.sum(torch.square(actions - previous_actions), dim=1)
                step_result = env.step(actions.detach())
                if len(step_result) == 5:
                    obs, _, _, _, _ = step_result
                else:
                    obs, _, _, _ = step_result
                previous_actions[:] = actions[:]

                if step % max(1, args.frame_stride) != 0:
                    continue

                filename = os.path.join(frames_dir, f"{saved_frames:06d}.png")
                _set_follow_camera(env, camera_env_id, camera_env, camera_handle)
                if env.device != "cpu":
                    env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                env.gym.write_camera_image_to_file(
                    env.sim,
                    camera_env,
                    camera_handle,
                    gymapi.IMAGE_COLOR,
                    filename,
                )

                lin_err = torch.norm(env.commands[:, :2] - env.base_lin_vel[:, :2], dim=1).mean().item()
                orient_err = torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1).mean().item()
                trace["time"].append(step * env.dt)
                trace["lin_vel_error"].append(lin_err)
                trace["orientation_error"].append(orient_err)
                trace["action_rate"].append(action_rate.mean().item())
                saved_frames += 1
                if saved_frames >= args.max_frames:
                    break

        metrics_path = os.path.join(_output_dir(args), "frames", item_name, "rollout_diagnostics.png")
        _plot_rollout_traces(trace, metrics_path, args.task, args.scenario)

        metadata_path = os.path.join(frames_dir, "metadata.txt")
        with open(metadata_path, "w") as f:
            f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"task: {args.task}\n")
            f.write(f"scenario: {args.scenario}\n")
            f.write(f"checkpoint: {load_path}\n")
            f.write(f"frames: {saved_frames}\n")
            f.write(f"frame_stride: {args.frame_stride}\n")
            f.write(f"camera_env: {camera_env_id}\n")
            f.write(f"diagnostics: {metrics_path}\n")

        print(f"Wrote frames: {frames_dir}")
        print(f"Wrote diagnostics: {metrics_path}")
        if exit_on_success:
            _exit_successfully_without_atexit()
        return frames_dir
    finally:
        _destroy_gym_resources(env, camera_env, camera_handle)


def _append_optional_args(cmd, args):
    pairs = (
        ("--num_envs", args.num_envs),
        ("--seed", args.seed),
        ("--load_run", args.load_run),
        ("--checkpoint", args.checkpoint),
        ("--output_dir", args.output_dir),
        ("--frame_width", args.frame_width),
        ("--frame_height", args.frame_height),
        ("--frame_stride", args.frame_stride),
        ("--max_frames", args.max_frames),
        ("--camera_env", args.camera_env),
    )
    for name, value in pairs:
        if value is not None:
            cmd.extend([name, str(value)])


def _run_child(args, task, scenario):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--viz_suite",
        "single",
        "--task",
        task,
        "--scenario",
        scenario,
        "--headless",
    ]
    _append_optional_args(cmd, args)
    subprocess.run(cmd, check=True)


def _suite_rollouts(args):
    ablations = [
        (name, scenario, (args.baseline_task, ablation_task, args.final_task))
        for name, scenario, ablation_task in ABLATION_TASKS
    ]
    final = [("mixed_stress", "mixed_stress", (args.baseline_task, args.final_task))]
    return ablations, final


def run_suite(args):
    out_dir = _output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "visualization_manifest.md")
    rollout_groups = []

    if args.viz_suite in ("terrain_gallery", "all"):
        terrain_paths = [write_terrain_overview(args, scenario) for scenario in SCENARIOS]
    else:
        terrain_paths = []

    ablation_rollouts, final_rollouts = _suite_rollouts(args)
    if args.viz_suite in ("ablations", "all"):
        rollout_groups.extend(ablation_rollouts)
    if args.viz_suite in ("final", "all"):
        rollout_groups.extend(final_rollouts)

    for _, scenario, tasks in rollout_groups:
        for task in tasks:
            _run_child(args, task, scenario)

    with open(manifest, "w") as f:
        f.write("# Robustness Visualization Manifest\n\n")
        f.write(f"- output_dir: `{out_dir}`\n")
        f.write(f"- max_frames_per_item: `{args.max_frames}`\n")
        f.write(f"- frame_stride: `{args.frame_stride}`\n\n")
        if terrain_paths:
            f.write("## Terrain overview figures\n\n")
            for path in terrain_paths:
                f.write(f"- `{path}`\n")
            f.write("\n")
        if rollout_groups:
            f.write("## Rollout frame folders\n\n")
            for _, scenario, tasks in rollout_groups:
                for task in tasks:
                    f.write(f"- `{os.path.join(out_dir, 'frames', f'{scenario}__{task}')}`\n")

    print(f"Wrote manifest: {manifest}")


def main():
    args = get_args()
    if args.viz_suite == "single":
        record_rollout_frames(args, exit_on_success=True)
        return
    if args.viz_suite not in ("terrain_gallery", "ablations", "final", "all"):
        raise ValueError("--viz_suite must be one of: single, terrain_gallery, ablations, final, all")
    run_suite(args)


if __name__ == "__main__":
    main()
