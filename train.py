import numpy as np
import torch
import torch.nn as nn
import pybullet_envs
import pybullet_envs.bullet as bul

import torch.nn.functional as F

import gym
import os

import time
from ee619.agent import Actor, Critic, ReplayBuffer, TD3
from collections import deque
import itertools as it

start_timestep=1e4

std_noise=0.02

env = gym.make('Walker2DBulletEnv-v0')

# Set seeds
seed = 12345
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state = env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])
threshold = env.spec.reward_threshold
threshold = 4000

print('start_dim: ', state_dim, ', action_dim: ', action_dim)
print('max_action: ', max_action, ', threshold: ', threshold, ', std_noise: ', std_noise)

agent = TD3(state_dim, action_dim, max_action)

def twin_ddd_train(n_episodes=100000, save_every=10, print_env=10):

    #scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []    

    time_start = time.time()                    # Init start time
    replay_buf = ReplayBuffer(state_dim, action_dim)                 # Init ReplayBuffer
    
    timestep_after_last_save = 0
    total_timesteps = 0
    
    low = env.action_space.low
    high = env.action_space.high
    
    print('Low in action space: ', low, ', High: ', high, ', Action_dim: ', action_dim)
            
    for i_episode in range(1, n_episodes+1):
        
        timestep = 0
        total_reward = 0
        
        # Reset environment
        state = env.reset()
        done = False
        
        while True:
            
            # Select action randomly or according to policy
            if total_timesteps < start_timestep:
                action = env.action_space.sample()
            else:
                action = agent.act(np.array(state))
                #if std_noise != 0: 
                shift_action = np.random.normal(0, std_noise, size=action_dim)
                action = (action + shift_action).clip(low, high)
            
            # Perform action
            next_state, reward, done, _ = env.step(action) 
            done_bool = 0 if timestep + 1 == env._max_episode_steps else float(done)
            total_reward += reward                          # full episode reward

            # Store every timestep in replay buffer
            replay_buf.add(state, action, next_state, reward, done_bool)
            state = next_state

            timestep += 1     
            total_timesteps += 1
            timestep_after_last_save += 1

            if done:                                       # done ?
                break                                      # save score

        #scores_deque.append(total_reward)
        scores_array.append(total_reward)

        #avg_score = np.mean(scores_deque)
        avg_score = np.mean(scores_array)
        avg_scores_array.append(avg_score)

        #max_score = np.max(scores_deque)
        max_score = np.max(scores_array)

        # train_by_episode(time_start, i_episode) 
        s = (int)(time.time() - time_start)
        if i_episode % print_env == 0 or (len(scores_array) == 100 and avg_score > threshold):
            print('Ep. {}, Timestep {},  Ep.Timesteps {}, Score: {:.2f}, Avg.Score: {:.2f}, Max.Score: {:.2f}, Time: {:02}:{:02}:{:02} '\
                .format(i_episode, total_timesteps, timestep, \
                        total_reward, avg_score, max_score, s//3600, s%3600//60, s%60))     

        agent.train(replay_buf, timestep)

        # Save episode if more than save_every=5000 timesteps
        if timestep_after_last_save >= save_every and i_episode > 0:

           timestep_after_last_save %= save_every            
           agent.save('chpnt_interm', 'dir_Walker2D_002')  
        
        if len(scores_deque) == 100 and avg_score >= threshold:
           print('Environment solved with Average Score: ',  avg_score )
           break 
            
    agent.save('chpnt_ts2500', 'dir_Walker2D_002')  

    return scores_array, avg_scores_array

if __name__ == '__main__':
    scores, avg_scores = twin_ddd_train()