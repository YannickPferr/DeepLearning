# -*- coding: utf-8 -*-
from ReplayBuffer import *
import numpy as np
import time

class DDQNAgent():
    
    def __init__(self, env, Q_net=None, target_net=None, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.999, epsilon_min=0.01, batch_size=32, replay_buffer_len=10000, pretraining_len=100,
                 target_update_freq=100, logging_freq=1, saving_freq=100, early_stop_last_100=None,
                 q_net_save_path='DDQNAgentQ_net.h5', target_net_save_path='DDQNAgenttarget_net.h5'):
        self.action_size = env.action_space.n
        self.replay_buffer = ReplayBuffer(size=replay_buffer_len)
        self.Q_net = Q_net
        self.target_net = target_net
        print(Q_net.summary())
        print(target_net.summary())
        self.env = env
        
        self.logging_freq = logging_freq
        self.saving_freq = saving_freq
        
        self.early_stop_last_100 = early_stop_last_100
        
        self.q_net_save_path = q_net_save_path
        self.target_net_save_path = target_net_save_path
        
        # hyperparameters
        self.gamma = gamma  # discount rate on future rewards
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay  # the decay of epsilon after each training batch
        self.epsilon_min = epsilon_min  # the minimum exploration rate permissible
        self.pretraining_len = pretraining_len
        self.batch_size = batch_size  # maximum size of the batches sampled from memory
        self.target_update_freq = target_update_freq
        self.rolling_avg_10 = 10
        self.rolling_avg_50 = 50
        self.rolling_avg_100 = 100
        self.rolling_avg_500 = 500
        
    def load_model(self):
        self.Q_net.load_weights(self.q_net_save_path)
        self.target_net.load_weights(self.target_net_save_path)
        
    def save_model(self):
        self.Q_net.save(self.q_net_save_path)
        self.target_net.save(self.target_net_save_path)
    
    def update_target_model(self):
        self.target_net.set_weights(self.Q_net.get_weights())
    
    #epsilon greedy
    def act(self, state, do_train=False):     
        if do_train and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        state = np.expand_dims(state, axis=0)
            
        q_vals = self.Q_net.predict(state)[0]
        return np.argmax(q_vals)
        
    def replay(self, batch_size):
        #get batch from replay buffer and collect every variable
        currStateBatch, actionBatch, rewardBatch, nextStateBatch, terminatedBatch = self.sample(batch_size)
            
        #calc qvals
        q_vals = self.Q_net.predict(currStateBatch)
        q_vals_next = self.Q_net.predict(nextStateBatch)
        target_next = self.target_net.predict(nextStateBatch)
        next_best_actions = np.argmax(q_vals_next, axis=1)
        q_vals[range(batch_size), actionBatch] = rewardBatch + (1 - terminatedBatch) * self.gamma * target_next[range(batch_size), next_best_actions]

        #decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        loss = self.Q_net.fit(currStateBatch, q_vals, verbose=0)
        return loss

    def store_exp(self, state, action, reward, next_state, terminated):
        self.replay_buffer.append((state, action, reward, next_state, terminated))
    
    def sample(self, batch_size):
        sample = self.replay_buffer.sample(batch_size)
        
        currStateBatch, actionBatch, rewardBatch, nextStateBatch, terminatedBatch = [],[],[],[],[]
        for s in sample:
            currStateBatch.append(s[0])
            actionBatch.append(s[1])
            rewardBatch.append(s[2])
            nextStateBatch.append(s[3])
            terminatedBatch.append(s[4])

        return np.array(currStateBatch), np.array(actionBatch), np.array(rewardBatch), np.array(nextStateBatch), np.array(terminatedBatch)

    def train(self, num_episodes, visualize=False):
        self.gather_exp(self.pretraining_len)
        self.run(num_episodes, do_train=True, visualize=visualize)
        
    def test(self, num_episodes, visualize=True):
        return self.run(num_episodes, visualize=visualize)

    def gather_exp(self, num_episodes):
        print("Start pretraining...")
        for e in range(1, num_episodes + 1):
            #reset environment
            state = self.env.reset()
            
            #init variables
            terminated = False
            while not terminated:
                #select action
                action = self.act(state, True)
                
                #take action and store experience in memory buffer   
                next_state, reward, terminated, info = self.env.step(action) 
                self.store_exp(state, action, reward, next_state, terminated)
                
                #set new state
                state = next_state
                    
        print("Pretraining done!")
                       
    
    #this is the main method that starts the training loop
    #at the start of the episode an initial observation is received from the environment
    #the agent then chooses an action and sends it to the environment
    #after execution the agent receives the next observation and the reward for his last action
    #one episode continues until the terminated state is received from the enviornment
    #during training the received rewards are logged and the model is contionusly being saved at fixed intervals
    def run(self, num_episodes, greedy=False, do_train=False, visualize=False, log=True, save=True, random=True):
        best = {}
        reward_list = []
        max_reward = -np.inf
        try:
            start = time.time()
            for e in range(1, num_episodes + 1):
                #reset environment
                observation = self.env.reset()
                if visualize:
                    self.env.render()
                
                #init variables
                episode_reward = 0
                terminated = False
                while not terminated:
                    state = observation
                 
                    #select action
                    action = self.act(state, do_train)
                    
                    #take action and store experience in memory buffer   
                    next_state, reward, terminated, info = self.env.step(action) 
                    self.store_exp(state, action, reward, next_state, terminated)
                    
                    #add reward and set new state
                    episode_reward += reward
                    observation = next_state
                    
                    #only replay memory if in training mode and enough samples have been collected
                    if do_train and len(self.replay_buffer) > self.batch_size:
                        self.replay(self.batch_size)
                       
                    if visualize:
                        self.env.render()
                
                reward_list.append(episode_reward)
                if episode_reward > max_reward:
                    max_reward = episode_reward
                    best = info
                if log and e % self.logging_freq == 0:
                    end = time.time()
                    last_n = 500 if 500 <= len(reward_list) else len(reward_list)
                    print("**********************************")
                    print("Episode: ", e, "/", num_episodes, " Time: ", round(end - start, 2), "s Reward: ", episode_reward, " Max Reward: ", max_reward," Mean (last 500): ", np.mean(reward_list[-last_n:]), sep="")   
                    start = time.time()
                  
                if do_train and e % self.target_update_freq == 0:    
                    self.update_target_model()    

                if save and e % self.saving_freq == 0:
                    self.save_model()    

                if self.early_stop_last_100 and e % 100 == 0 and np.mean(reward_list[-last_n:]) >= self.early_stop_last_100:
                    break  
        except KeyboardInterrupt:
            print("Keyboard interrupt!")
        
        num_finished_epsiodes = len(reward_list) 
        
        if save:
            self.save_model() 
       
        if log:
            print("Finished", num_finished_epsiodes, "Epsiodes! Max Reward:", np.max(reward_list), "Mean Reward (last 500):", np.mean(reward_list[-last_n:]))
        self.reward_list = reward_list 
        self.env.close()
  
        return best, max_reward, reward_list

        