# DeepLearning
Deep Learning repo with implementations for REINFORCE policy gradient algorithm and a Deep Double Q-Learning Network (DDQN). Models were tested using the OpenAI Gym environments CartPole-v1 and LunarLander-v2.

The REINFORCE agent (with or without a baseline) can be trained much faster than the DDQN because the DDQN is much heavier on computation. Though it is sometimes said to learn faster in terms of the number of training episodes. In my tests however, the REINFORCE agent also turned out to be the faster learner when it comes to the amount of episodes needed to achieve good performance.

#### REINFORCE (with baseline) agent in CartPole-v1 after training for 1000 episodes:

![156061767-29093c73-c906-464a-97fe-aa5cea04f451](https://user-images.githubusercontent.com/37211050/156062349-6de493c5-4bcb-4a2f-93a1-d1d743a1d49b.gif)


## How to run
1. Clone the repo 
2. Install the needed packages tensorflow/keras, gym, numpy (you can try with requirements.txt just note that it was created with a M1 Mac)
3. Change configs/hyperparameters as needed in ToyProblemsTrain.py
4. Run ToyProblemsTrain.py

