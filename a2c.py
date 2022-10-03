import retro
import torch
import numpy as np
from collections import deque
import math
import os

from algos.agents import A2CAgent
from algos.models import ActorCnn, CriticCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, "custom_integrations")
)

env = retro.make("BattleCity-Nes", inttype=retro.data.Integrations.ALL)
env.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
env.reset()

possible_actions = {
    # No Operation
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Up
    1: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    # Down
    2: [0, 0, 0, 0, 0, 1, 0, 0, 0],
    # Left
    3: [0, 0, 0, 0, 0, 0, 1, 0, 0],
    # Right
    4: [0, 0, 0, 0, 0, 0, 0, 1, 0],
    # a
    5: [1, 0, 0, 0, 0, 0, 0, 0, 0],
}

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (1, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = len(possible_actions)
SEED = 0
GAMMA = 0.99           # discount factor
ALPHA= 0.0001          # Actor learning rate
BETA = 0.0005          # Critic learning rate
UPDATE_EVERY = 100     # how often to update the network

agent = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)


def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0

        # Punish the agent for not moving forward
        prev_state = {}
        steps_stuck = 0
        timestamp = 0
        while timestamp < 10000:
            action, log_prob, entropy = agent.act(state)
            next_state, reward, done, info = env.step(possible_actions[action])
            env.render()
            score += reward

            timestamp += 1
            # Punish the agent for standing still for too long.
            if (prev_state == info):
                steps_stuck += 1
            else:
                steps_stuck = 0
            prev_state = info

            if (steps_stuck > 20):
                reward -= 1

            next_state = stack_frames(state, next_state, False)
            agent.step(state, log_prob, entropy, reward, done, next_state)
            state = next_state
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

    return scores


def main():

    agent = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
    agent.load_model(SCRIPT_DIR + '/models/a2c')

    env.viewer = None
    # watch an untrained agent
    state = stack_frames(None, env.reset(), True)
    for j in range(10000):
        env.render(close=False)
        action, _, _ = agent.act(state)
        next_state, reward, done, _ = env.step(possible_actions[action])
        state = stack_frames(state, next_state, False)
        if done:
            env.reset()
            break
    env.render(close=True)


if __name__ == "__main__":
    #scores = train(300)
    #agent.save_model('models/a2c')
    main()
