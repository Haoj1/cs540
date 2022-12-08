import gym
import random
import numpy as np
import time
from collections import deque
import pickle

from collections import defaultdict

EPISODES = 20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0


if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)

    Q_table = defaultdict(default_Q_value)  # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0

        # TODO PERFORM Q LEARNING
        # using done as flag
        done = False
        # initialize the first state
        obs = env.reset()
        # do until end
        while not done:
            # Epsilon-greedy policy
            # make random action
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            # select a best action in this state
            else:
                predict = []
                for x in range(env.action_space.n):
                    predict.append(Q_table[(obs, x)])
                predict = np.array(predict)
                action = np.argmax(predict)
            # make action
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            # update q value
            q = Q_table[(obs, action)]
            # when not done, using the first algorithm
            if not done:
                max_q = np.max(np.array([Q_table[(next_obs, x)] for x in range(env.action_space.n)]))
                Q_table[(obs, action)] = q + LEARNING_RATE * \
                                             (reward + DISCOUNT_FACTOR * max_q - q)
                # old_q = Q_table[(pre_obs, action)]
                # td = reward + (DISCOUNT_FACTOR * np.max(
                #     np.array([Q_table[(obs, action)] for action in range(env.action_space.n)]))) - Q_table[
                #          (pre_obs, action)]
                # Q_table[(pre_obs, action)] = old_q + (LEARNING_RATE * td)
            # when terminate using another
            else:
                Q_table[(obs, action)] = q + LEARNING_RATE * (reward - q)
            # update the state
            obs = next_obs
        # update the reward
        episode_reward_record.append(episode_reward)
        EPSILON *= EPSILON_DECAY

        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record)) / 100))
            print("EPSILON: " + str(EPSILON))

    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl', 'wb')
    pickle.dump([Q_table, EPSILON], model_file)
    #######################
