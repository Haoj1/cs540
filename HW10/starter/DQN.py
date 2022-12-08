import torch.nn as nn
import torch.nn.functional as functional
from collections import deque
import gym
import random
import torch
import numpy as np
import torch.optim as optim
import time
import pickle


EARLY_STOPPING_THRESHOLD = 80 # we stop training and immediately save our model when we reach a this average score over the past 100 episodes
INPUT_DIMENSIONS = 4
OUTPUT_DIMENSIONS = 2
MAX_QUEUE_LENGTH = 1000000
EPSILON = 1
EPSILON_DECAY = .999
MIN_EPSILON = .05
EPOCHS =  2000
DISCOUNT_FACTOR = 0.995
TARGET_NETWORK_UPDATE_FREQUENCY = 5000
MINI_BATCH_SIZE = 32
PRETRAINING_LENGTH = 1000





class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIMENSIONS,12)
        self.fc2 = nn.Linear(12,12)
        self.fc3 = nn.Linear (12, OUTPUT_DIMENSIONS)


    def forward(self,x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ExperienceReplayBuffer():

    def __init__(self):
        #TODO complete ExperienceReplayBuffer __init__
        #Depends on MAX_QUEUE_LENGTH
        #HINT: use a deque object
        self.buffer = deque(maxlen=MAX_QUEUE_LENGTH)


    def sample_mini_batch(self):
        #TODO complete ExperienceReplayBuffer sample_mini_batch
        #Depends on MINI_BATCH_SIZE
        # random.seed(10)
        samples = random.sample(self.buffer, k=MINI_BATCH_SIZE)
        return samples

    def append(self,experience):
        #TODO complete ExperienceReplayBuffer append
        self.buffer.append(experience)


if __name__ == "__main__":

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    policy_net = Network()
    target_policy_net = Network()

    target_policy_net.load_state_dict(policy_net.state_dict()) # here we update the target policy network to match the policy network


    env = gym.envs.make("CartPole-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)


    queue = ExperienceReplayBuffer()

    optimizer = optim.Adam(policy_net.parameters(), lr=.001)

    step_counter = 0

    episode_reward_record = deque(maxlen=100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(EPOCHS):
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
           
            #TODO collect experience sample and add to experience replay buffer
            with torch.no_grad():
                if random.uniform(0, 1) < EPSILON:
                    action = env.action_space.sample()
                else:
                    argument = torch.tensor(obs).to(torch.float32)
                    outputs = policy_net(argument)
                    action = torch.argmax(outputs)

                    action = int(action)
                    # print(action)
                obs_, reward, done, info = env.step(action)
                # if done:
                #     print(episode_reward)
                episode_reward += reward
                queue.append((obs, action, reward, obs_, done))
                obs = obs_
            if step_counter >= PRETRAINING_LENGTH:
                experience = queue.sample_mini_batch()

                #TODO Fill in the missing code to perform a Q update on 'policy_net'
                #for the sampled experience minibatch 'experience'
                # states = []
                # states_ = []
                # rewards = []
                # actions = []
                # dones = []
                # # print(experience[0])
                # for item in experience:
                #     states.append(item[0])
                #     states_.append(item[3])
                #     rewards.append(item[2])
                #     actions.append(item[1])
                #     dones.append(item[4])
                # states = torch.tensor(states).to(torch.float32)
                # states_ = torch.tensor(states_).to(torch.float32)
                # rewards = torch.tensor(rewards).to(torch.float32)
                # actions = torch.tensor(actions).to(torch.int64)
                # indices = np.arange(MINI_BATCH_SIZE)
                # # print(actions)
                # q_eval = policy_net(states)[indices, actions]
                # q_next = target_policy_net(states_).max(dim=1)[0]
                # # print(q_eval)
                # # print(actions)
                # estimate = q_eval.reshape(-1)
                # # print(estimate.shape)
                # # y = torch.zeros((MINI_BATCH_SIZE, 1))
                # y = rewards + DISCOUNT_FACTOR * q_next
                # # print(estimate.shape)
                # y_vector = y.reshape(-1)
                # # y_vector = rewards + DISCOUNT_FACTOR * q_next
                # loss = functional.smooth_l1_loss(estimate,y_vector)
                batch_states = []
                batch_rewards = []
                for j in range(MINI_BATCH_SIZE):
                    current_experience = experience[j]  # Contains (s, a, r, s', done)
                    batch_states.append(current_experience[0])  # Add state to  all states
                    with torch.no_grad():
                        current_estimate = policy_net(torch.tensor(np.array(current_experience[0])).float()).detach()
                    if current_experience[4] is True:  # If current state is at end
                        current_estimate[current_experience[1]] = current_experience[2]  # Set Q[action] = reward
                    else:
                        with torch.no_grad():
                            Q_sp_ap_prediction = np.array(
                                target_policy_net(torch.tensor(np.array(current_experience[3])).float()))
                            action_sp_ap = np.argmax(Q_sp_ap_prediction)
                        current_estimate[current_experience[1]] = current_experience[2] + DISCOUNT_FACTOR * \
                                                                  Q_sp_ap_prediction[action_sp_ap]
                    batch_rewards.append(current_estimate)

                batch_states_tensor = torch.FloatTensor(batch_states)
                batch_estimate = policy_net(batch_states_tensor)
                batch_rewards_tensor = torch.clone(batch_estimate).detach()

                for j in range(MINI_BATCH_SIZE):
                    batch_rewards_tensor[j] = batch_rewards[j]

                loss = functional.smooth_l1_loss(batch_estimate, batch_rewards_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_counter % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                target_policy_net.load_state_dict(policy_net.state_dict()) # here we update the target policy network to match the policy network
            step_counter += 1
        episode_reward_record.append(episode_reward)
        EPSILON = EPSILON * EPSILON_DECAY
        if EPSILON < MIN_EPSILON:
            EPSILON = MIN_EPSILON

        if i%100 ==0 and i>0:
            last_100_avg = sum(list(episode_reward_record))/100
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(last_100_avg))
            print("EPSILON: " +  str(EPSILON))
            if last_100_avg > EARLY_STOPPING_THRESHOLD:
                break

    
    torch.save(policy_net.state_dict(), "DQN.mdl")
    pickle.dump([EPSILON], open("DQN_DATA.pkl",'wb'))






