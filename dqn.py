import retro
import torch
import numpy as np
from collections import deque
import math
import os

from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
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
    1: [1, 0, 0, 0, 1, 0, 0, 0, 0],
    # Down
    2: [1, 0, 0, 0, 0, 1, 0, 0, 0],
    # Left
    3: [1, 0, 0, 0, 0, 0, 1, 0, 0],
    # Right
    4: [1, 0, 0, 0, 0, 0, 0, 1, 0],
    # a
    5: [1, 0, 0, 0, 0, 0, 0, 0, 0],
    # Up
    6: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    # Down
    7: [0, 0, 0, 0, 0, 1, 0, 0, 0],
    # Left
    8: [0, 0, 0, 0, 0, 0, 1, 0, 0],
    # Right
    9: [0, 0, 0, 0, 0, 0, 0, 1, 0],
}


def random_play():
    score = 0
    env.reset()

    for i in range(2000):
        env.render()
        action = possible_actions[np.random.randint(len(possible_actions))]
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            print("Your Score at end of game is: ", score)
            break
    env.reset()
    env.render(close=True)


def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (1, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = len(possible_actions)
SEED = 0
GAMMA = 0.5          # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 32        # Update batch size
LR = 0.0001            # learning rate
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 300     # how often to update the network
UPDATE_TARGET = 10000  # After which thershold replay to be started
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.001         # Ending value of epsilon
EPS_DECAY = 100         # Rate by which epsilon to be decayed

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx / EPS_DECAY)


def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)

        # Punish the agent for not moving forward
        prev_state = 3
        timestamp = 0

        while timestamp < 10000:
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(possible_actions[action])
            env.render()
            score = reward

            timestamp += 1

            # Punish the agent for standing still for too long.
            if (info['Lives'] < prev_state) and prev_state != 0:
                prev_state -= 1
                reward -= 1000

            if done:
                reward -= 1000000

            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.6f}'.format(i_episode, np.mean(scores_window), eps),
              end="")

    return scores


def main():
    env.viewer = None
    # watch an untrained agent
    state = stack_frames(None, env.reset(), True)
    for j in range(10000):
        env.render(close=False)
        action = agent.act(state, eps=0.91)
        next_state, reward, done, _ = env.step(possible_actions[action])
        state = stack_frames(state, next_state, False)
        if done:
            env.reset()
            break
    env.render(close=True)


if __name__ == "__main__":
    scores = train(1000)
    main()
