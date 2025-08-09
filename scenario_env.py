import numpy as np
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle, vehicle_class_to_type
from agents.random_policy import RandomPolicy
import math
import logging
from collections import defaultdict
from typing import Union, Dict, AnyStr
from metadrive.engine.logger import get_logger, set_log_level

class PolicyVehicle(DefaultVehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = None

    def set_policy(self, policy):
        self.policy = policy

    def act(self, observation):
        if self.policy is not None:
            return self.policy.act(observation)
        else:
            return self.action_space.sample()

    def before_step(self, action):
        steering, throttle = action
        self._action = {
            "steering": float(steering),
            "throttle": float(throttle)
        }

vehicle_class_to_type[PolicyVehicle] = "default"

class MultiAgentScenarioEnv(ScenarioEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(dict(
            data_directory=None,
            num_controlled_agents=3,
            horizon=1000,
        ))
        return config

    def __init__(self, config, agent2policy=None):
        self.num_controlled_agents = config.get("num_controlled_agents", 3)
        self.agent2policy = agent2policy or {}
        self.controlled_agents = {}
        self.controlled_agent_ids = []
        super().__init__(config)

    def _get_all_obs(self):
        obs = {}
        for agent_id, vehicle in self.controlled_agents.items():
            state = vehicle.get_state()
            obs[agent_id] = {
                "position": state["position"],
                "velocity": state["velocity"],
            }
        return obs

    def reset(self, seed: Union[None, int] = None):
        if self.logger is None:
            self.logger = get_logger()
            log_level = self.config.get("log_level", logging.DEBUG if self.config.get("debug", False) else logging.INFO)
            set_log_level(log_level)

        self.lazy_init()
        self._reset_global_seed(seed)
        if self.engine is None:
            raise ValueError("Broken MetaDrive instance.")

        self.engine.reset()
        self.reset_sensors()
        self.engine.taskMgr.step()
        if self.top_down_renderer is not None:
            self.top_down_renderer.clear()
            self.engine.top_down_renderer = None

        self.dones = {}
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        self.controlled_agents.clear()
        self.controlled_agent_ids.clear()

        super().reset(seed)  # 初始化场景
        self._spawn_controlled_agents()

        return self._get_all_obs()

    def _spawn_controlled_agents(self):
        current_map = self.engine.current_map
        lane_ids = list(current_map.road_network.graph.keys())

        ego_vehicle = self.engine.agent_manager.active_agents.get("default_agent")
        ego_position = ego_vehicle.position if ego_vehicle else np.array([0, 0])

        for i in range(self.num_controlled_agents):
            agent_id = f"controlled_{i}"
            nearby_lanes = []
            for lane_id in lane_ids:
                lane = current_map.road_network.get_lane(lane_id)
                start_pos = lane.position(0, 0)
                dist = np.linalg.norm(np.array(start_pos) - np.array(ego_position))
                if dist < 50.0:
                    nearby_lanes.append(lane_id)
            if not nearby_lanes:
                nearby_lanes = lane_ids

            lane_id = self.np_random.choice(nearby_lanes)
            lane = current_map.road_network.get_lane(lane_id)
            s = float(self.np_random.uniform(0, lane.length))
            p1 = lane.position(s, 0.0)
            x1, y1 = float(p1[0]), float(p1[1])
            eps = 1.0
            s2 = s + eps if s + eps < lane.length else s - eps
            p2 = lane.position(s2, 0.0)
            x2, y2 = float(p2[0]), float(p2[1])
            heading = math.atan2(y2 - y1, x2 - x1)

            vehicle = self.engine.spawn_object(
                PolicyVehicle,
                vehicle_config={},
                position=(x1, y1),
                heading=heading
            )
            vehicle.reset(position=(x1, y1), heading=heading)

            policy = self.agent2policy.get(agent_id, RandomPolicy())
            print(f"[INFO] {agent_id} bound to {policy.__class__.__name__}")
            vehicle.set_policy(policy)

            self.controlled_agents[agent_id] = vehicle
            self.controlled_agent_ids.append(agent_id)

            # ✅ 关键：注册到引擎的 active_agents，才能参与物理更新
            self.engine.agent_manager.active_agents[agent_id] = vehicle

    def step(self, action_dict: Dict[AnyStr, Union[list, np.ndarray]]):
        for agent_id, action in action_dict.items():
            if agent_id in self.controlled_agents:
                self.controlled_agents[agent_id].before_step(action)

        self.engine.step()

        for agent_id in action_dict:
            if agent_id in self.controlled_agents:
                self.controlled_agents[agent_id].after_step()

        obs = self._get_all_obs()
        rewards = {aid: 0.0 for aid in self.controlled_agents}
        dones = {aid: False for aid in self.controlled_agents}
        dones["__all__"] = self.episode_step >= self.config["horizon"]
        infos = {aid: {} for aid in self.controlled_agents}
        return obs, rewards, dones, infos
