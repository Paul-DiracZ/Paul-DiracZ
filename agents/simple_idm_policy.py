import numpy as np

class ConstantVelocityPolicy:
    def __init__(self, target_speed=50):
        self.target_speed = target_speed

    def act(self, obs):
        # obs["velocity"] 应该是一个二元向量 [vx, vy]
        velocity = np.array(obs["velocity"], dtype=np.float32)
        speed = np.linalg.norm(velocity)  # 当前速度大小

        # 简单比例控制：当 speed < target_speed 时加油，否则松油
        # 输出范围 [0, 1]
        throttle = (self.target_speed - speed) / self.target_speed
        throttle = float(np.clip(throttle, 0.0, 1.0))

        steering = 0.0  # 始终直行

        return [steering, throttle]
