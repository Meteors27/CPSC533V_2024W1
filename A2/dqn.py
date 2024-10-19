from pydoc import render_doc
import re
import gymnasium as gym
import math
import random
from itertools import count
import torch
from eval_policy import eval_policy, device
from model import MyModel
from replay_buffer import ReplayBuffer
import numpy as np
import torch.nn.functional as F
import ipdb


BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 10
NUM_EPISODES = 4000
TEST_INTERVAL = 100
LEARNING_RATE = 20e-4
RENDER_INTERVAL = 20
ENV_NAME = "CartPole-v1"
PRINT_INTERVAL = 10

env = gym.make(ENV_NAME, render_mode="human")
state_shape = len(env.reset()[0])
n_actions = env.action_space.n

model = MyModel(state_shape, n_actions).to(device)
# load the best model if it exists
try:
    model.load_state_dict(torch.load("best_model_{}.pt".format(ENV_NAME)))
    print("loading model.")
except FileNotFoundError:
    print("training model from scratch.")

target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer()


def choose_action(state, test_mode=False):
    if not test_mode and random.random() < EPS_EXPLORATION:
        return torch.tensor(env.action_space.sample(), device=device).view(1, 1)
    else:
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return model.select_action(state)


def optimize_model(state, action, next_state, reward, done):
    # TODO given a tuple (s_t, a_t, s_{t+1}, r_t, done_t) update your model weights
    memory.push(state, action, reward, next_state, done)
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    actions = actions.long()
    # ipdb.set_trace()
    y_i = reward + (1 - dones) * GAMMA * target(next_states).max(1)[0].detach()
    loss = F.mse_loss(y_i.unsqueeze(1), model(states).gather(1, actions))

    # state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    # next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(
    #     0
    # )
    # y_i = reward + (1 - done) * GAMMA * target(next_state).max(1)[0].detach()
    # loss = F.mse_loss(y_i.unsqueeze(1), model(state).gather(1, action))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_reinforcement_learning(render=False):
    steps_done = 0
    best_score = -float("inf")

    for i_episode in range(1, NUM_EPISODES + 1):
        episode_total_reward = 0
        state, _ = env.reset()
        for t in count():
            action = choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(
                action.cpu().numpy()[0][0]
            )
            steps_done += 1
            episode_total_reward += reward

            optimize_model(state, action, next_state, reward, terminated)

            state = next_state

            if render:
                env.render()

            if terminated or truncated:
                if i_episode % PRINT_INTERVAL == 0:
                    print(
                        "[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]".format(
                            i_episode, NUM_EPISODES, t, episode_total_reward
                        )
                    )
                break

        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

        if i_episode % TEST_INTERVAL == 0:
            print("-" * 10)
            score = eval_policy(policy=model, env=ENV_NAME, render=render)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), "best_model_{}.pt".format(ENV_NAME))
                print("saving model.")
            print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
            print("-" * 10)


if __name__ == "__main__":
    train_reinforcement_learning(render=True)
