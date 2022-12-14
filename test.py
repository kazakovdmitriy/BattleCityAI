import numpy as np
import retro
import os
import random


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integrations")
    )

    env = retro.make("BattleCity-Nes", inttype=retro.data.Integrations.ALL)
    env.seed(0)
    obs = env.reset()

    actions = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    while True:

        obs, rew, done, info = env.step(random.choice(actions))
        env.render()

        if done:
            print(f"Total reward: {rew}")
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
