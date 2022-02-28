# -*- coding: utf-8 -*-
SEED = 42

from numpy.random import seed
seed(SEED)
from tensorflow.random import set_seed
set_seed(SEED)
from random import seed
seed(SEED)

from DDQNAgent import *
from REINFORCEAgent import *

import sys
import gym

from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

#CONFIGS
#choose "CartPole-v1" or "LunarLander-v2" (others may work too, but models were tested with these envs)
env_str = "CartPole-v1"
env = gym.make(env_str)
env.seed(SEED)
#cartpole is considered solved at a score of 475 and lunar lander at a score of 200 over the last 100 episodes
if env_str == "CartPole-v1":  
    early_stop_last_100 = 475
elif env_str == "LunarLander-v2":  
    early_stop_last_100 = 200
else:
    early_stop_last_100 = None
    print("Non tested env selected, no early stop score will be used!")

#choose one of "DDQN", "REINFORCE", "REINFORCE_BASELINE"
model = "REINFORCE_BASELINE"
train = True

EPISODES_TO_TRAIN = 10000
EPISODES_TO_TEST = 3
        
stateSize = env.observation_space.sample().shape
numActions = env.action_space.n
numHiddenUnitsDDQN = 64
numHiddenLayersDDQN = 2
numHiddenUnitsREINFORCE = 64
numHiddenLayersREINFORCE = 2
lr = 0.001
seed_init = GlorotUniform(seed=SEED)

print("Train config:", "Environment:", env_str, " | Model:", model, " | Train:", train, " | Train Episodes:", EPISODES_TO_TRAIN, " | Test Episodes:", EPISODES_TO_TEST)
#DDQN
if model == "DDQN":
    input = Input(shape=stateSize)
    dense = Dense(numHiddenUnitsDDQN, activation="relu", kernel_initializer=seed_init)(input)
    for i in range(numHiddenLayersDDQN - 1):
        dense = Dense(numHiddenUnitsDDQN, activation="relu", kernel_initializer=seed_init)(dense)
    action_logits = Dense(numActions, activation="linear", kernel_initializer=seed_init)(dense)
    Q_net = Model(input, action_logits)
    Q_net.compile(loss='mse', optimizer=Adam(learning_rate=lr))

    target_net = models.clone_model(Q_net)

    agent = DDQNAgent(env, Q_net=Q_net, target_net=target_net, early_stop_last_100=early_stop_last_100, target_update_freq=10, gamma=1, q_net_save_path='models/'+env_str+'QNet.h5', target_net_save_path='models/'+env_str+'TargetNet.h5')
elif model == "REINFORCE":
    #REINFORCE
    input = Input(shape=stateSize)
    dense = Dense(numHiddenUnitsREINFORCE, activation="relu", kernel_initializer=seed_init)(input)
    for i in range(numHiddenLayersREINFORCE - 1):
        dense = Dense(numHiddenUnitsREINFORCE, activation="relu", kernel_initializer=seed_init)(dense)
    output = Dense(numActions, activation="softmax", kernel_initializer=seed_init)(dense)

    model = Model(inputs=[input], outputs=[output])
    model.compile(optimizer=Adam(learning_rate=lr))

    agent = REINFORCEAgent(env=env, policy=model, early_stop_last_100=early_stop_last_100, policy_save_path='models/'+env_str+'REINFORCE.h5')
elif model == "REINFORCE_BASELINE":
    #BASELINE REINFORCE
    input = Input(shape=stateSize)
    dense = Dense(numHiddenUnitsREINFORCE, activation="relu", kernel_initializer=seed_init)(input)
    for i in range(numHiddenLayersREINFORCE - 1):
        dense = Dense(numHiddenUnitsREINFORCE, activation="relu", kernel_initializer=seed_init)(dense)
    output = Dense(numActions, activation="softmax", kernel_initializer=seed_init)(dense)

    model = Model(inputs=[input], outputs=[output])
    model.compile(optimizer=Adam(learning_rate=lr))

    input = Input(shape=stateSize)
    dense = Dense(numHiddenUnitsREINFORCE, activation="relu", kernel_initializer=seed_init)(input)
    for i in range(numHiddenLayersREINFORCE - 1):
        dense = Dense(numHiddenUnitsREINFORCE, activation="relu", kernel_initializer=seed_init)(dense)
    action_logits = Dense(1, activation="linear", kernel_initializer=seed_init)(dense)

    value = Model(inputs=[input], outputs=[action_logits])
    value.compile(optimizer=Adam(learning_rate=lr), loss="mse")

    agent = REINFORCEAgent(env=env, policy=model, value=value, baseline=True, early_stop_last_100=early_stop_last_100,
                                policy_save_path='models/'+env_str+'PolicyREINFORCEwBaseline.h5',  
                                value_save_path='models/'+env_str+'ValueREINFORCEwBaseline.h5')
else:
    print("No valid model selected!")
    sys.exit(0)

if train:
    agent.train(EPISODES_TO_TRAIN)
else:
    agent.load_model()
agent.test(EPISODES_TO_TEST)

