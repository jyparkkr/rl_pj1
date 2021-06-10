"""Train an agent on a Walker2DBullet environment."""
from os.path import abspath, dirname, realpath, join
import time
from collections import deque

import gym
from gym import logger
# pybullet_envs must be imported in order to create Walker2DBulletEnv
import pybullet_envs    # noqa: F401  # pylint: disable=unused-import
import numpy as np
import torch
from ee619.agent import Agent
from ee619.replay_memory import ReplayMemory


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory


def save(agent, episode, reward):
    """Save state of agent.

    Args:
        agent: The agent on train.
        episode: Number of episode on current training step.
        reward: Reward on current training step.
    """

    policy = join(ROOT, 'saved_model', f'weights_policy_{episode:05}_{reward:5.2f}.pth')
    critic = join(ROOT, 'saved_model', f'weights_critic_{episode:05}_{reward:5.2f}.pth')
    if episode == 'final':
        policy = join(ROOT, 'saved_model', 'weights_policy_final.pth')
        critic = join(ROOT, 'saved_model', 'weights_critic_final.pth')

    torch.save(agent.policy.state_dict(), policy)
    torch.save(agent.critic.state_dict(), critic)


def train(env, agent: Agent, max_episodes: int, threshold: int, max_steps: int, seed: int):
    """Computes the mean episodic return of the agent.

    Args:
        agent: The agent to evaluate.
        max_episodes: Number of maximum episode to stop.
	    threshold: Threshold of average score to stop.
        max_steps: Number of maximum step for every episode.
        seed: Passed to the environment for determinism.
    """
    logger.set_level(logger.DISABLED)
    total_numsteps = 0
    updates = 0
    num_episodes = 20000
    updates=0

    time_start = time.time()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = [] 

    batch_size=256 ## Training batch size
    start_steps=10000 ## Steps sampling random actions
    replay_size=1000000 ## size of replay buffer
    memory = ReplayMemory(seed, replay_size)


    
    for i_episode in range(num_episodes): 
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        for step in range(max_steps):    
            if start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > batch_size:                
                # Update parameters of all the networks
                agent.update_parameters(memory, batch_size, updates)

                updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            
            if done:
                break

        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)        
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        max_score = np.max(scores_deque)
        
        if i_episode % 100 == 0 and i_episode > 0:
            reward_round = round(episode_reward, 2)
            save(agent, i_episode, reward_round)

        s =  (int)(time.time() - time_start)
            
        print("Ep.: {}, Total Steps: {}, Ep.Steps: {}, Score: {:.3f}, Avg.Score: {:.3f}, Max.Score: {:.3f}, Time: {:02}:{:02}:{:02}".\
            format(i_episode, total_numsteps, episode_steps, episode_reward, avg_score, max_score, \
                  s//3600, s%3600//60, s%60))

                    
        if (avg_score > threshold):
            print('Solved environment with Avg Score:  ', avg_score)
            save(agent, 'final')
            break;
            
    return scores_array, avg_scores_array 


if __name__ == '__main__':
    seed = 0  
    env = gym.make('Walker2DBulletEnv-v0')

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    env.seed(seed)
    max_steps = env._max_episode_steps # 1000

    train(env=env, agent=Agent(), max_episodes=20000, threshold=2500, max_steps=max_steps, seed=seed)
