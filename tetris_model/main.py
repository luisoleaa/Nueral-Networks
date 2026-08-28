import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import random
import time
from model import Layer, Relu, softmax, gradient_check, full_forward
from grid import get_state_vector


env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)



state = get_state_vector(env)



num_actions = 12

Layer1 = Layer(238, 128)
Layer2 = Layer(128, 64)

Relu1 = Relu()
Relu2 = Relu()

policy_head = Layer(64, num_actions) # outputs one score per action
value_head  = Layer(64, 1)             # outputs one scalar

out1 = Layer1.forward(state)
Relu1_output = Relu1.forward(out1)
out2 = Layer2.forward(Relu1_output)
Relu2_output = Relu2.forward(out2)

action_logits = policy_head.forward(Relu2_output)   # shape (1, 12)
state_value = value_head.forward(Relu2_output) 

action_probs = softmax(action_logits) # shape (1,12), sums to 1 for whole probability
action = np.random.choice(num_actions, p = action_probs[0]) # picking 1 action based on probability 

print(action_probs)
print(action)

done = True
for step in range(10000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(action)
    
    env.render()
    # new_reward = info['score']
    # print(new_reward)
 
    # if step % 1000 == 0:
    #     time.sleep(5)
    # if step % 7000 == 0:
    #         time.sleep(15)

env.close()