import numpy as np

class RandomPolicy:
    def act(self, obs):
        # 动作 = [油门（加速）[-1~1], 转向角[-1~1]]
        return np.random.uniform(-1, 1, size=(2,))
