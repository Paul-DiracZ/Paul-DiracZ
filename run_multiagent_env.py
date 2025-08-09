from scenario_env import MultiAgentScenarioEnv
from agents.random_policy import RandomPolicy
from agents.simple_idm_policy import ConstantVelocityPolicy
from metadrive.engine.asset_loader import AssetLoader

WAYMO_DATA_DIR = r"C:\Users\Paul\PycharmProjects\Multi_Agent_Env\exp_converted"

if __name__ == "__main__":
    agent2policy = {
        "controlled_0": ConstantVelocityPolicy(target_speed=50),
        "controlled_1": ConstantVelocityPolicy(target_speed=50),
        "controlled_2": ConstantVelocityPolicy(target_speed=50)
    }

    env = MultiAgentScenarioEnv(
        config={
            "data_directory": AssetLoader.file_path(AssetLoader.asset_path, "waymo", unix_style=False),
            "num_controlled_agents": 3,  # 可以改成 3 看多车效果
            "horizon": 300,
            "use_render": True,
            "sequential_seed": True,
            "reactive_traffic": True,
            "manual_control": True,
        },
        agent2policy=agent2policy
    )

    obs = env.reset(2)
    for step in range(2000):
        actions = {
            aid: env.controlled_agents[aid].policy.act(obs[aid])
            for aid in env.controlled_agents
        }

        # 打印策略动作
        for aid, action in actions.items():
            print(f"Step {step} | {aid} action: {action}")

        obs, rewards, dones, infos = env.step(actions)
        env.render()

        # ✅ 打印车辆位置和速度，方便验证是否移动
        for aid, ob in obs.items():
            pos = ob["position"]
            vel = ob["velocity"]
            print(f"  {aid}: pos=({pos[0]:.2f}, {pos[1]:.2f}), vel=({vel[0]:.2f}, {vel[1]:.2f})")

        if dones["__all__"]:
            break

    env.close()
