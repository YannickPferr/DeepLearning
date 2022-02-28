# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import time

#Agent that implements the vanilla REINFORCE Policy Gradient algorithm
#optionally a learned baseline can be used to reduce the variance
class REINFORCEAgent():
    
    def __init__(self, env, policy=None, value=None, gamma=0.99, baseline=False,
                 logging_freq=1, saving_freq=100, early_stop_last_100=None,
                 policy_save_path='REINFORCEAgentPolicy.h5', value_save_path='REINFORCEAgentValue.h5'):
        
        self.action_size = env.action_space.n
        
        self.env = env
        self.policy = policy
        self.value = value
        
        self.early_stop_last_100 = early_stop_last_100

        self.logging_freq = logging_freq
        self.saving_freq = saving_freq
        
        self.policy_save_path = policy_save_path
        self.value_save_path = value_save_path
        
        # hyperparameters
        self.gamma = gamma  # discount rate on future rewards
        
        self.baseline = baseline
        
        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

    #load model from disk
    def load_model(self):
        self.policy.load_weights(self.policy_save_path)
        if self.baseline and self.value:
            self.value.load_weights(self.value_save_path)
    
    #save model to disk
    def save_model(self):
        self.policy.save(self.policy_save_path)
        if self.value is not None:
            self.value.save(self.value_save_path)
    
    #calculate discounted rewards
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float64)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
            
        return discounted_rewards
    
    #select next action either greedy or based on probability distribution
    def act(self, state, greedy=False):     
        probs = self.policy(np.expand_dims(state, axis=0)).numpy()[0]
    
        if not greedy:
            return np.random.choice(self.action_size, p=probs)
        
        return np.argmax(probs)
    
    #main function that the agent needs to learn from his experiences
    #the function updates the policy gradient
    #if a baseline is used, the value net predicts the likely reward and subtracts it from the actual reward
    #to reduce the variance of the vanilla REINFORCE algorithm
    def update(self):
        self.states = np.array([state for state in self.states])
        
        #discount and normalize rewards
        discounted_rewards = self.discount_rewards(self.rewards)
        value = 0
        #subtract baseline
        if self.baseline:
            #train value net
            self.value.train_on_batch(self.states, discounted_rewards)
            #value prediction as baseline
            value = self.value.predict(self.states).flatten()
        
        advantage = discounted_rewards - value
        #update policy gradients
        with tf.GradientTape() as tape:
            predictions = self.policy(np.array(self.states))
            predictions = K.clip(predictions, 1e-7, 1-1e-7)
            one_hot_values = tf.one_hot(np.array(self.actions), self.action_size)
            action_prob = K.sum(one_hot_values * predictions, axis=1)
            log_action_prob = K.log(action_prob)
            loss = -K.sum(log_action_prob * advantage)
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        
        #clear memory
        self.states, self.actions, self.rewards = [], [], []
        return loss, advantage
    
    #method to store sates, rewards and actions as experience
    def store_exp(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    #greedily chooses next action (used for inference after training)
    def test(self, num_episodes, visualize=True):
        return self.run(num_episodes, greedy=True, visualize=visualize, save=False, random=True)

    #start training
    def train(self, num_episodes, visualize=False):
        self.run(num_episodes, do_train=True, visualize=visualize)
            
    #this is the main method that starts the training loop
    #at the start of the episode an initial observation is received from the environment
    #the agent then chooses an action and sends it to the environment
    #after execution the agent receives the next observation and the reward for his last action
    #one episode continues until the terminated state is received from the enviornment
    #during training the received rewards are logged and the model is contionusly being saved at fixed intervals
    def run(self, num_episodes, greedy=False, do_train=False, visualize=False, log=True, save=True, random=True):
        best = {}
        loss_list = []
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
                    action = self.act(state, greedy)
        
                    #take action and store experience in memory buffer   
                    next_observation, reward, terminated, info = self.env.step(action)
                    self.store_exp(state, action, reward)
                        
                    #add reward and set new state
                    episode_reward += reward
                    
                    observation = next_observation
                       
                    if visualize:
                        self.env.render()
                
                #only replay memory if in training mode
                if do_train:
                    loss, reward = self.update()
                    loss_list.append(loss)  
                
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
  
        return best, max_reward, reward_list, loss_list