from collections import deque
import gym
import random
import numpy as np
import time
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

    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
       
        # TODO perform SARSA learning

        done_1 = False
        done_2 = False
        obs_2 = env.reset()
        while not done_1 and not done_2:

            old_obs = obs_2

            # Epsilon-greedy policy 1
            if random.uniform(0, 1) < EPSILON:
                action_1 = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(old_obs, i)] for i in range(env.action_space.n)])
                action_1 = np.argmax(prediction)

            # Take action 1
            obs_1, reward_1, done_1, info_1 = env.step(action_1)
            episode_reward += reward_1

            # Epsilon-greedy policy 2
            if random.uniform(0, 1) < EPSILON:
                action_2 = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs_1, i)] for i in range(env.action_space.n)])
                action_2 = np.argmax(prediction)

            # Take action 2
            obs_2, reward_2, done_2, info_2 = env.step(action_2)
            episode_reward += reward_2

            # Update rules
            old_q = Q_table[(old_obs, action_1)]
            td = reward_1 + (DISCOUNT_FACTOR * Q_table[(obs_1, action_2)]) - Q_table[(old_obs, action_1)]
            Q_table[(old_obs, action_1)] = old_q + (LEARNING_RATE * td)

            if done_1 or done_2:
                curr_q = Q_table[(obs_1, action_2)]
                td = reward_2 - Q_table[(obs_1, action_2)]
                Q_table[(obs_1, action_2)] = curr_q + (LEARNING_RATE * td)

        episode_reward_record.append(episode_reward)
        EPSILON = EPSILON * EPSILON_DECAY

        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################



