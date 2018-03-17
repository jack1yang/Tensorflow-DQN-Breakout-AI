import gym
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from Agent import Agent_RL

env = gym.make('BreakoutDeterministic-v4')
agent = Agent_RL(restore=True)


def pre_process(state):
    state = np.uint8(resize(rgb2gray(state), (84, 84), mode='reflect'))
    return state


def play():
    done = 0
    frame = env.reset()
    total_reward = 0

    history = np.zeros([84, 84, 5], dtype=np.uint8)
    for i in range(4):
        history[:, :, i] = pre_process(frame)

    while not done:
        env.render()
        action = agent.policy(history[:, :, :4], epsilon=0.05)
        frame, reward, done, info = env.step(action)
        history[:, :, 4] = pre_process(frame)
        total_reward += reward
        history[:, :, :4] = history[:, :, 1:]

    print("Score: %d" % total_reward)


if __name__ == "__main__":
    play()
