from collections import deque
import gym
import random
import numpy as np
import time
import pickle

from collections import defaultdict


EPISODES =   20000
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
       
        #TODO perform SARSA learning

        # initialize the first state and the flag
        obs = env.reset()
        done = False
        # take actions immediately using greedy algorithm
        if random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(np.array(Q_table[(obs, x)] for x in range(env.action_space.n)))
        # do until terminated
        while not done:
            # take action and get next state
            next_obs, reward, done, info = env.step(action)

            episode_reward += reward
            # get next action
            if random.uniform(0, 1) < EPSILON:
                next_action = env.action_space.sample()
            else:
                # prediction = np.array([Q_table[(next_obs, x)] for x in range(env.action_space.n)])
                next_action = np.argmax(np.array([Q_table[(next_obs, x)] for x in range(env.action_space.n)]))
                # next_action = np.argmax(prediction)
            # always update the current Q value
            q = Q_table[(obs, action)]
            q_target = Q_table[(next_obs, next_action)]
            Q_table[(obs, action)] = q + LEARNING_RATE * \
                                         (reward + DISCOUNT_FACTOR * q_target - q)
            # when terminate update the final valuw
            if done:
                Q_table[(next_obs, next_action)] = q_target + LEARNING_RATE * \
                                                   (reward - q_target)
            # update the state and action
            obs = next_obs
            action = next_action
        episode_reward_record.append(episode_reward)
        EPSILON = EPSILON * EPSILON_DECAY

        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )


    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################



