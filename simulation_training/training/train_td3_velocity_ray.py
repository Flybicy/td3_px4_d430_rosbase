import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange


@dataclass
class Obstacle:
    center: np.ndarray
    radius: float
    height: float
    kind: str = "cylinder"


class VelocityActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 3, max_speed: float = 1.5):
        super().__init__()
        self.max_speed = float(max_speed)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.max_speed


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 3):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_states[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


class Planner3DEnv:
    def __init__(
        self,
        world_size: float = 20.0,
        max_speed: float = 1.5,
        dt: float = 0.05,
        max_z_vel: float = 0.25,
        yaw_rays: int = 12,
        pitch_rays: int = 3,
        sensor_range: float = 5.0,
        z_min: float = 0.3,
        z_max: float = 2.8,
        goal_radius: float = 0.35,
        collision_margin: float = 0.08,
        target_same_altitude: bool = True,
        altitude_hold_weight: float = 0.35,
        vertical_speed_weight: float = 0.45,
        action_smooth_weight: float = 0.18,
        vertical_smooth_weight: float = 0.45,
        ring_count_min: int = 1,
        ring_count_max: int = 4,
        ring_segments: int = 12,
    ):
        self.world_size = world_size
        self.max_speed = max_speed
        self.dt = dt
        self.max_z_vel = max_z_vel
        self.yaw_rays = yaw_rays
        self.pitch_rays = pitch_rays
        self.sensor_range = sensor_range
        self.z_min = z_min
        self.z_max = z_max
        self.goal_radius = goal_radius
        self.collision_margin = collision_margin
        self.target_same_altitude = target_same_altitude
        self.altitude_hold_weight = altitude_hold_weight
        self.vertical_speed_weight = vertical_speed_weight
        self.action_smooth_weight = action_smooth_weight
        self.vertical_smooth_weight = vertical_smooth_weight
        self.ring_count_min = ring_count_min
        self.ring_count_max = ring_count_max
        self.ring_segments = ring_segments
        self.ray_dirs = self._make_ray_directions(yaw_rays, pitch_rays)
        self.ray_count = len(self.ray_dirs)
        self.state_dim = 10 + self.ray_count
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.obstacles: List[Obstacle] = []

    def reset(self):
        self.position = self._sample_free_point()
        self.goal = self._sample_goal(self.position)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.obstacles = self._sample_obstacles()
        return self.build_observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action[2] = np.clip(action[2], -self.max_z_vel, self.max_z_vel)
        speed = float(np.linalg.norm(action))
        if speed > self.max_speed:
            action = action / (speed + 1e-6) * self.max_speed

        old_distance = float(np.linalg.norm(self.goal[:2] - self.position[:2]))
        self.velocity = action
        self.position = self.position + self.velocity * self.dt
        new_distance = float(np.linalg.norm(self.goal[:2] - self.position[:2]))
        altitude_error = abs(float(self.position[2] - self.goal[2]))

        reached = new_distance < self.goal_radius and altitude_error < 0.25
        collided = self._in_collision(self.position)
        out_of_bounds = bool(
            abs(self.position[0]) > self.world_size
            or abs(self.position[1]) > self.world_size
            or self.position[2] < self.z_min
            or self.position[2] > self.z_max
        )

        progress = old_distance - new_distance
        nearest_clearance = self._nearest_clearance(self.position)
        obstacle_penalty = 0.0
        if nearest_clearance < 0.8:
            obstacle_penalty = -0.2 * (0.8 - nearest_clearance)

        action_delta = action - self.prev_action
        reward = 8.0 * progress - 0.02
        reward += obstacle_penalty
        reward -= 0.01 * float(np.linalg.norm(action) ** 2)
        reward -= self.action_smooth_weight * float(np.linalg.norm(action_delta) ** 2)
        reward -= self.vertical_smooth_weight * float((action[2] - self.prev_action[2]) ** 2)
        reward -= self.vertical_speed_weight * float(action[2] ** 2)
        reward -= self.altitude_hold_weight * altitude_error

        done = reached or collided or out_of_bounds
        if reached:
            reward += 20.0
        if collided:
            reward -= 25.0
        if out_of_bounds:
            reward -= 15.0

        self.prev_action = action.copy()

        return self.build_observation(), float(reward), done, {
            "reached": reached,
            "collided": collided,
            "out_of_bounds": out_of_bounds,
            "distance": new_distance,
        }

    def build_observation(self):
        goal_vec = self.goal - self.position
        goal_dist = float(np.linalg.norm(goal_vec))
        obs = [
            *(self.position / self.world_size),
            *(self.velocity / self.max_speed),
            *(goal_vec / self.world_size),
            goal_dist / self.world_size,
        ]
        obs.extend(self._ray_distances() / self.sensor_range)
        return np.asarray(obs, dtype=np.float32)

    def _make_ray_directions(self, yaw_rays: int, pitch_rays: int):
        pitch_min = math.radians(-45.0)
        pitch_max = math.radians(45.0)
        pitches = np.linspace(pitch_min, pitch_max, pitch_rays)
        yaws = np.linspace(-math.pi, math.pi, yaw_rays, endpoint=False)
        dirs = []
        for pitch in pitches:
            cp = math.cos(float(pitch))
            sp = math.sin(float(pitch))
            for yaw in yaws:
                dirs.append(np.array([
                    cp * math.cos(float(yaw)),
                    cp * math.sin(float(yaw)),
                    sp,
                ], dtype=np.float32))
        return np.asarray(dirs, dtype=np.float32)

    def _ray_distances(self):
        distances = np.full(self.ray_count, self.sensor_range, dtype=np.float32)
        for ray_idx, ray_dir in enumerate(self.ray_dirs):
            best = self.sensor_range
            for obstacle in self.obstacles:
                hit = self._ray_obstacle_intersection(self.position, ray_dir, obstacle)
                if hit is not None:
                    best = min(best, hit)
            distances[ray_idx] = best
        return distances

    def _ray_obstacle_intersection(self, origin, direction, obstacle):
        if obstacle.kind == "sphere":
            return self._ray_sphere_intersection(origin, direction, obstacle)
        return self._ray_cylinder_intersection(origin, direction, obstacle)

    def _ray_cylinder_intersection(self, origin, direction, obstacle):
        dx, dy = float(direction[0]), float(direction[1])
        ox = float(origin[0] - obstacle.center[0])
        oy = float(origin[1] - obstacle.center[1])
        a = dx * dx + dy * dy
        if a < 1e-6:
            return None

        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - obstacle.radius * obstacle.radius
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None

        sqrt_disc = math.sqrt(disc)
        candidates = []
        for t in ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)):
            if 0.0 <= t <= self.sensor_range:
                z = float(origin[2] + t * direction[2])
                if 0.0 <= z <= obstacle.height:
                    candidates.append(t)

        if not candidates:
            return None
        return float(min(candidates))

    def _ray_sphere_intersection(self, origin, direction, obstacle):
        oc = origin - obstacle.center
        b = 2.0 * float(np.dot(oc, direction))
        c = float(np.dot(oc, oc) - obstacle.radius * obstacle.radius)
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) * 0.5
        t2 = (-b + sqrt_disc) * 0.5
        candidates = [t for t in (t1, t2) if 0.0 <= t <= self.sensor_range]
        if not candidates:
            return None
        return float(min(candidates))

    def _sample_free_point(self, min_distance_from=None, min_distance=0.0, xy_distance=False):
        for _ in range(1000):
            point = np.array([
                np.random.uniform(-self.world_size * 0.8, self.world_size * 0.8),
                np.random.uniform(-self.world_size * 0.8, self.world_size * 0.8),
                np.random.uniform(self.z_min, self.z_max),
            ], dtype=np.float32)
            if min_distance_from is not None:
                delta = point[:2] - min_distance_from[:2] if xy_distance else point - min_distance_from
                if np.linalg.norm(delta) < min_distance:
                    continue
            return point
        return np.array([0.0, 0.0, (self.z_min + self.z_max) * 0.5], dtype=np.float32)

    def _sample_goal(self, start):
        for _ in range(1000):
            goal = self._sample_free_point(min_distance_from=start, min_distance=5.0, xy_distance=True)
            if self.target_same_altitude:
                goal[2] = np.clip(
                    start[2] + np.random.normal(0.0, 0.15),
                    self.z_min,
                    self.z_max,
                )
            return goal
        return start.copy()

    def _sample_obstacles(self):
        count = random.randint(8, 16)
        obstacles = []
        for _ in range(count):
            center = np.array([
                np.random.uniform(-self.world_size * 0.75, self.world_size * 0.75),
                np.random.uniform(-self.world_size * 0.75, self.world_size * 0.75),
                0.0,
            ], dtype=np.float32)
            radius = random.uniform(0.35, 0.9)
            height = random.uniform(self.z_max + 0.2, self.z_max + 0.8)
            if np.linalg.norm(center[:2] - self.position[:2]) < radius + 1.0:
                continue
            if np.linalg.norm(center[:2] - self.goal[:2]) < radius + 1.0:
                continue
            obstacles.append(Obstacle(center=center, radius=radius, height=height))
        obstacles.extend(self._sample_ring_obstacles())
        return obstacles

    def _sample_ring_obstacles(self):
        rings = []
        ring_count = random.randint(self.ring_count_min, self.ring_count_max)
        for _ in range(ring_count):
            center = np.array([
                np.random.uniform(-self.world_size * 0.65, self.world_size * 0.65),
                np.random.uniform(-self.world_size * 0.65, self.world_size * 0.65),
                np.random.uniform(0.8, 1.4),
            ], dtype=np.float32)
            if np.linalg.norm(center[:2] - self.position[:2]) < 1.5:
                continue
            if np.linalg.norm(center[:2] - self.goal[:2]) < 1.5:
                continue

            yaw = np.random.uniform(-math.pi, math.pi)
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            rot = np.array([
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            radius_y = np.random.uniform(0.65, 1.0)
            radius_z = np.random.uniform(0.55, 0.9)
            tube_radius = np.random.uniform(0.08, 0.14)

            for i in range(self.ring_segments):
                angle = 2.0 * math.pi * i / self.ring_segments
                local = np.array([
                    0.0,
                    radius_y * math.cos(angle),
                    radius_z * math.sin(angle),
                ], dtype=np.float32)
                bead_center = center + rot.dot(local)
                if self.z_min <= bead_center[2] <= self.z_max:
                    rings.append(Obstacle(
                        center=bead_center,
                        radius=tube_radius,
                        height=0.0,
                        kind="sphere",
                    ))
        return rings

    def _in_collision(self, point):
        return any(
            self._point_in_obstacle(point, obstacle)
            for obstacle in self.obstacles
        )

    def _point_in_obstacle(self, point, obstacle):
        if obstacle.kind == "sphere":
            return np.linalg.norm(point - obstacle.center) <= obstacle.radius + self.collision_margin
        return (
            np.linalg.norm(point[:2] - obstacle.center[:2]) <= obstacle.radius + self.collision_margin
            and 0.0 <= point[2] <= obstacle.height
        )

    def _nearest_clearance(self, point):
        if not self.obstacles:
            return self.sensor_range
        return min(
            self._obstacle_clearance(point, obstacle)
            for obstacle in self.obstacles
        )

    def _obstacle_clearance(self, point, obstacle):
        if obstacle.kind == "sphere":
            return float(np.linalg.norm(point - obstacle.center) - obstacle.radius)
        return float(np.linalg.norm(point[:2] - obstacle.center[:2]) - obstacle.radius)


def soft_update(source: nn.Module, target: nn.Module, tau: float):
    with torch.no_grad():
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(src_param.data, alpha=tau)


def hard_update(source: nn.Module, target: nn.Module):
    target.load_state_dict(source.state_dict())


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = Planner3DEnv(
        world_size=args.world_size,
        max_speed=args.max_speed,
        dt=args.dt,
        max_z_vel=args.max_z_vel,
        yaw_rays=args.yaw_rays,
        pitch_rays=args.pitch_rays,
        sensor_range=args.sensor_range,
        z_min=args.z_min,
        z_max=args.z_max,
        target_same_altitude=args.target_same_altitude,
        altitude_hold_weight=args.altitude_hold_weight,
        vertical_speed_weight=args.vertical_speed_weight,
        action_smooth_weight=args.action_smooth_weight,
        vertical_smooth_weight=args.vertical_smooth_weight,
        ring_count_min=args.ring_count_min,
        ring_count_max=args.ring_count_max,
        ring_segments=args.ring_segments,
    )

    actor = VelocityActor(env.state_dim, max_speed=args.max_speed).to(device)
    actor_target = VelocityActor(env.state_dim, max_speed=args.max_speed).to(device)
    critic = Critic(env.state_dim).to(device)
    critic_target = Critic(env.state_dim).to(device)
    hard_update(actor, actor_target)
    hard_update(critic, critic_target)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = ReplayBuffer(env.state_dim, 3, args.buffer_size)

    os.makedirs(args.save_dir, exist_ok=True)
    save_metadata(args, env)
    rewards_path = os.path.join(args.save_dir, "rewards.csv")
    best_eval = -math.inf
    total_steps = 0

    with open(rewards_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "reached", "collided", "out_of_bounds", "final_distance"])

        progress = trange(args.episodes, desc="training")
        for episode in progress:
            state = env.reset()
            episode_reward = 0.0
            info = {"reached": False, "collided": False, "out_of_bounds": False, "distance": 0.0}

            for step in range(args.max_steps):
                total_steps += 1
                if total_steps < args.start_steps:
                    action = np.random.uniform(-args.max_speed, args.max_speed, size=3).astype(np.float32)
                else:
                    with torch.no_grad():
                        state_t = torch.as_tensor(state, device=device).unsqueeze(0)
                        action = actor(state_t).cpu().numpy()[0]
                    noise = np.random.normal(0, args.exploration_noise, size=3).astype(np.float32)
                    action = np.clip(action + noise, -args.max_speed, args.max_speed)

                next_state, reward, done, info = env.step(action)
                replay.add(state, action, reward, next_state, float(done))
                state = next_state
                episode_reward += reward

                if replay.size >= args.batch_size:
                    update_td3(actor, actor_target, critic, critic_target,
                               actor_opt, critic_opt, replay, device, total_steps, args)

                if done:
                    break

            writer.writerow([
                episode,
                f"{episode_reward:.4f}",
                step + 1,
                int(info["reached"]),
                int(info["collided"]),
                int(info["out_of_bounds"]),
                f"{info['distance']:.4f}",
            ])
            f.flush()

            if episode % args.eval_interval == 0 or episode == args.episodes - 1:
                eval_score = evaluate(actor, env, device, args.eval_episodes, args.max_steps)
                if eval_score > best_eval:
                    best_eval = eval_score
                    export_actor(actor, env.state_dim, args.save_dir, device)
                    export_actor_npz(actor, args.save_dir)
                progress.set_postfix(
                    reward=f"{episode_reward:.1f}",
                    eval=f"{eval_score:.1f}",
                    best=f"{best_eval:.1f}",
                )

    export_actor(actor, env.state_dim, args.save_dir, device)
    export_actor_npz(actor, args.save_dir)
    torch.save(actor.state_dict(), os.path.join(args.save_dir, "actor_state_dict.pt"))
    print(f"Saved TorchScript actor to: {os.path.join(args.save_dir, 'actor.pt')}")
    print(f"Saved NumPy actor to: {os.path.join(args.save_dir, 'actor_weights.npz')}")
    print(f"State dimension: {env.state_dim}")


def update_td3(actor, actor_target, critic, critic_target, actor_opt, critic_opt, replay, device, total_steps, args):
    state, action, reward, next_state, done = replay.sample(args.batch_size, device)

    with torch.no_grad():
        noise = torch.randn_like(action) * args.policy_noise
        noise = noise.clamp(-args.noise_clip, args.noise_clip)
        next_action = (actor_target(next_state) + noise).clamp(-args.max_speed, args.max_speed)
        target_q1, target_q2 = critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target = reward + (1.0 - done) * args.gamma * target_q

    current_q1, current_q2 = critic(state, action)
    critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    if total_steps % args.policy_delay == 0:
        actor_loss = -critic.q1_value(state, actor(state)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        soft_update(actor, actor_target, args.tau)
        soft_update(critic, critic_target, args.tau)


def evaluate(actor, env, device, episodes, max_steps):
    actor.eval()
    scores = []
    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            total = 0.0
            for _ in range(max_steps):
                action = actor(torch.as_tensor(state, device=device).unsqueeze(0)).cpu().numpy()[0]
                state, reward, done, _ = env.step(action)
                total += reward
                if done:
                    break
            scores.append(total)
    actor.train()
    return float(np.mean(scores))


def export_actor(actor, state_dim, save_dir, device):
    actor.eval()
    example = torch.zeros(1, state_dim, device=device)
    traced = torch.jit.trace(actor, example)
    traced.save(os.path.join(save_dir, "actor.pt"))
    actor.train()


def export_actor_npz(actor, save_dir):
    actor.eval()
    state = actor.state_dict()
    np.savez(
        os.path.join(save_dir, "actor_weights.npz"),
        w0=state["net.0.weight"].detach().cpu().numpy(),
        b0=state["net.0.bias"].detach().cpu().numpy(),
        w1=state["net.2.weight"].detach().cpu().numpy(),
        b1=state["net.2.bias"].detach().cpu().numpy(),
        w2=state["net.4.weight"].detach().cpu().numpy(),
        b2=state["net.4.bias"].detach().cpu().numpy(),
        max_speed=np.asarray(actor.max_speed, dtype=np.float32),
    )
    actor.train()


def save_metadata(args, env):
    metadata = {
        "state_dim": env.state_dim,
        "action_dim": 3,
        "world_size": args.world_size,
        "max_speed": args.max_speed,
        "dt": args.dt,
        "max_z_vel": args.max_z_vel,
        "yaw_rays": args.yaw_rays,
        "pitch_rays": args.pitch_rays,
        "sensor_range": args.sensor_range,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "target_same_altitude": args.target_same_altitude,
        "altitude_hold_weight": args.altitude_hold_weight,
        "vertical_speed_weight": args.vertical_speed_weight,
        "action_smooth_weight": args.action_smooth_weight,
        "vertical_smooth_weight": args.vertical_smooth_weight,
        "ring_count_min": args.ring_count_min,
        "ring_count_max": args.ring_count_max,
        "ring_segments": args.ring_segments,
    }
    with open(os.path.join(args.save_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-dir", type=str, default="runs/td3_velocity_ray_arm")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--world-size", type=float, default=20.0)
    parser.add_argument("--max-speed", type=float, default=1.5)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-z-vel", type=float, default=0.25)
    parser.add_argument("--yaw-rays", type=int, default=12)
    parser.add_argument("--pitch-rays", type=int, default=3)
    parser.add_argument("--sensor-range", type=float, default=5.0)
    parser.add_argument("--z-min", type=float, default=0.3)
    parser.add_argument("--z-max", type=float, default=2.8)
    parser.add_argument("--target-same-altitude", action="store_true", default=True)
    parser.add_argument("--altitude-hold-weight", type=float, default=0.35)
    parser.add_argument("--vertical-speed-weight", type=float, default=0.45)
    parser.add_argument("--action-smooth-weight", type=float, default=0.18)
    parser.add_argument("--vertical-smooth-weight", type=float, default=0.45)
    parser.add_argument("--ring-count-min", type=int, default=1)
    parser.add_argument("--ring-count-max", type=int, default=4)
    parser.add_argument("--ring-segments", type=int, default=12)
    parser.add_argument("--buffer-size", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--start-steps", type=int, default=3000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--exploration-noise", type=float, default=0.35)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
