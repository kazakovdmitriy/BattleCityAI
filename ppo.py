import os
import retro
import torch
import numpy as np
from collections import deque

from algos.agents import PPOAgent
from algos.models import ActorCnn, CriticCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, "custom_integrations")
)

env = retro.make("BattleCity-Nes", inttype=retro.data.Integrations.ALL)
env.seed(0)

# if gpu is to be used
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
BETA = 0.0001          # Critic learning rate
TAU = 0.95
BATCH_SIZE = 32
PPO_EPOCH = 15
CLIP_PARAM = 0.2
UPDATE_EVERY = 300     # how often to update the network


agent = PPOAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn, CriticCnn)

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

        while True:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, info = env.step(possible_actions[action])
            env.render()
            score += reward

            if done:
                reward = -1000

            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, value, log_prob, reward, done, next_state)
            if done:
                break
            else:
                state = next_state

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

    return scores


def main():
    env.viewer = None

    agent = PPOAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn, CriticCnn)
    agent.load_model(SCRIPT_DIR + '/models/ppo')

    state = stack_frames(None, env.reset(), True)
    for j in range(1000):
        env.render(close=False)
        action, _, _ = agent.act(state)
        next_state, reward, done, _ = env.step(possible_actions[action])
        state = stack_frames(state, next_state, False)
        if done:
            env.reset()
            break
    env.render(close=True)


if __name__ == "__main__":
    scores = train(500)
    # main()