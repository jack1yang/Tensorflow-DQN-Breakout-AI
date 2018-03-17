import shutil
import pickle
import random
import copy
import numpy as np
import gym
from skimage.transform import resize
from skimage.color import rgb2gray
from Agent import Agent_RL
from collections import deque

restore = True

# input/output
input_size = 84, 84
history_size = 4
output_size = 4

# replay memory
batch_size = 32
replay_memory_size = 500000
replay_start_size = 50000

# training parameter
discount_factor = 0.99
initial_epsilon = 1
final_epsilon = 0.1
final_exploration_frame = 1000000
target_update_frequency = 10000
update_frequency = 4
no_op_max = 30

env = gym.make('BreakoutDeterministic-v4')
agent = Agent_RL(restore=restore)


def save_data(batch, epsilon, episode, step, data_episode, data_avg_reward, data_max_reward, data_q_avg):
    with open("./data/batch.dat", 'wb') as f:
        pickle.dump(batch, f)
    with open("./data/parameter.dat", 'wb') as f:
        pickle.dump((epsilon, episode, step, data_episode, data_avg_reward, data_max_reward, data_q_avg), f)
    shutil.copy("./data/batch.dat", "./data/batch_backup.dat")
    shutil.copy("./data/parameter.dat", "./data/parameter_backup.dat")
    print("Batch, Parameter saved in file: ./data/batch.dat")


def restore_data():
    with open("./data/batch.dat", 'rb') as f:
        try:
            batch = pickle.load(f)
        except EOFError:
            with open("./data/batch_backup.dat") as fb:
                batch = pickle.load(fb)
            with open("./data/parameter_backup.dat") as fb:
                epsilon, episode, step, data_episode, data_avg_reward, data_max_reward, data_q_avg = pickle.load(fb)
            print("Batch, Parameter are restored.")
            return batch, epsilon, episode, step, data_episode, data_avg_reward, data_max_reward, data_q_avg

    with open("./data/parameter.dat", 'rb') as f:
        epsilon, episode, step, data_episode, data_avg_reward, data_max_reward, data_q_avg = pickle.load(f)
    print("Batch, Parameter are restored.")
    return batch, epsilon, episode, step, data_episode, data_avg_reward, data_max_reward, data_q_avg


def pre_process(state):
    state = np.uint8(resize(rgb2gray(state), (84, 84), mode='reflect'))
    return state


# state = history[:, :, :4], next_state = history[:, :, 1:5]
def update(agent, batch, step, dis=discount_factor):
    x_stack = np.empty(0).reshape([0, 84, 84, 4])
    y_stack = np.empty(0).reshape([0, 4])

    for history, action, reward, done in batch:
        state = history[:, :, :4]
        next_state = history[:, :, 1:]

        Q = agent.Q_main.predict(state)

        if not done:
            # target greedy policy
            next_action_value = np.max(agent.Q_target.predict(next_state))
            Q[0, action] = reward + dis * next_action_value
        else:
            Q[0, action] = reward

        _Q = copy.deepcopy(Q)
        _state = copy.deepcopy(state)
        _state = np.reshape(_state, [1, 84, 84, 4])

        y_stack = np.vstack([y_stack, _Q])
        x_stack = np.vstack([x_stack, _state])

    agent.Q_main.update(x_stack, y_stack, step)


def train():
    if restore:
        replay_memory, epsilon, episode, step, data_episode, data_avg_reward, data_max_reward, data_q_avg = restore_data()
    else:
        replay_memory = deque()
        epsilon = initial_epsilon
        episode = 0
        step = 0
        data_episode = []
        data_avg_reward = []
        data_max_reward = []
        data_q_avg = []

    while True:
        episode += 1
        done = 0
        frame = env.reset()
        check = 0
        life = 5

        # history_size = 4
        history = np.zeros([84, 84, 5], dtype=np.uint8)
        for i in range(history_size):
            history[:, :, i] = pre_process(frame)

        # state = history[:, :, :4], next_state = history[:, :, 1:5]
        while not done:
            step += 1
            epsilon = max(final_epsilon, epsilon - (0.9 / final_exploration_frame))
            action = agent.policy(history[:, :, :4], epsilon=epsilon)

            # check no_op_max
            if action == 0 or action == 1:
                check += 1
            else:
                check = 0
            if check > no_op_max:
                action = random.choice([2, 3])
                check = 0

            frame, reward, done, info = env.step(action)

            _done = done
            if life != info['ale.lives']:
                life = info['ale.lives']
                _done = True

            history[:, :, 4] = pre_process(frame)
            data = copy.deepcopy((history, action, reward, _done))

            # store transition (s, a, r, s', d) in memory
            replay_memory.append(data)
            if len(replay_memory) > replay_memory_size:
                replay_memory.popleft()

            # backup history
            history[:, :, :4] = history[:, :, 1:]

            # update main network
            if len(replay_memory) >= replay_start_size:
                if step % update_frequency == 0:
                    mini_batch = random.sample(replay_memory, batch_size)
                    update(agent, mini_batch, step)

            # reset target network
            if step % target_update_frequency == 0:
                agent.copy()
                print("Target Network is updated.")

        if len(replay_memory) >= replay_start_size:
            if episode % 10 == 0:
                total_reward = 0
                maximum_reward = 0
                Q_avg = 0.0

                for _ in range(10):
                    done = 0
                    frame = env.reset()
                    episode_reward = 0
                    Q_sum = 0.0
                    Q_step = 0

                    history = np.zeros([84, 84, 5], dtype=np.uint8)
                    for i in range(4):
                        history[:, :, i] = pre_process(frame)

                    while not done:
                        action = agent.policy(history[:, :, :4], epsilon=0.05)

                        Q = agent.Q_main.predict(history[:, :, :4])
                        Q_sum += Q[0, action]
                        Q_step += 1

                        frame, reward, done, _ = env.step(action)
                        history[:, :, 4] = pre_process(frame)
                        episode_reward += reward
                        history[:, :, :4] = history[:, :, 1:]

                    maximum_reward = max(maximum_reward, episode_reward)
                    total_reward += episode_reward
                    Q_avg += (Q_sum / Q_step)

                Q_avg /= 10.0
                print("[Episode %d] Steps: %d, Mean Reward: %.1f, Max Reward: %d, Q Value: %.5f, Replay Mem Size: %d, Epsilon: %.5f, Learning Rate: %.9f"
                      % (episode, step, total_reward / 10.0,
                         maximum_reward, Q_avg, len(replay_memory),
                         epsilon, agent.learning_rate(step)))

                # append data
                if episode % 1000 == 0:
                    data_episode.append(episode)
                    data_avg_reward.append(total_reward / 10.0)
                    data_max_reward.append(maximum_reward)
                    data_q_avg.append(Q_avg)

                # save data
                if episode % 100 == 0:
                    save_data(replay_memory,
                              epsilon,
                              episode,
                              step,
                              data_episode,
                              data_avg_reward,
                              data_max_reward,
                              data_q_avg)
                    agent.save()
        elif episode % 10 == 0:
            print("[Episode %d] Steps: %d, Replay Mem Size: %d, Epsilon: %.5f, Learning Rate: %.9f"
                  % (episode, step, len(replay_memory),
                     epsilon, agent.learning_rate(step)))


if __name__ == "__main__":
    train()
