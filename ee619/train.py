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
import matplotlib.pyplot as plt
import pickle



ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory


def save(agent, episode, reward):
    """Save state of agent.

    Args:
        agent: The agent on train.
        episode: Number of episode on current training step.
        reward: Reward on current training step.
    """

    policy = join(ROOT, 'saved_model', f'weights_policy_ep_{episode:05}_rw_{reward:5.2f}.pth')
    critic = join(ROOT, 'saved_model', f'weights_critic_ep_{episode:05}_rw_{reward:5.2f}.pth')
    if episode == 'final':
        policy = join(ROOT, 'saved_model', 'weights_policy_final.pth')
        critic = join(ROOT, 'saved_model', 'weights_critic_final.pth')

    torch.save(agent.policy.state_dict(), policy)
    torch.save(agent.critic.state_dict(), critic)


def save_score_plot(scores, avg_scores, std_scores, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores, label="Score")
    plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, label="Avg on 100 episodes")
    plt.fill_between(
        np.arange(1, len(avg_scores)+1), 
        avg_scores - std_scores, 
        avg_scores + std_scores, 
        color='gray', 
        alpha=0.2
    )

    plt.legend() 
    plt.ylabel('Score')
    plt.xlabel('Episodes #')
    if name is not None:
        path = join(ROOT, 'train_result', 'score_' + str(avg_scores[-1])[:7]+'.png')
        plt.savefig(path)

    plt.clf()


def save_loss_plot(losses, name):
    #import matplotlib.pyplot as plt

    #print('length of losses: ', len(losses))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(losses)+1), losses, label= name)
    #plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, label="Avg on 100 episodes")
    plt.legend(bbox_to_anchor=(1.05, 1)) 
    plt.ylabel('Loss')
    plt.xlabel('Episodes #')
    
    path = join(ROOT, 'train_result', name + '.png')
    plt.savefig(path)

    plt.clf()


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
    updates=0

    time_start = time.time()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = [] 
    std_scores_array = []

    qf1_loss_array = []
    qf2_loss_array = []
    policy_loss_array = []
    alpha_loss_array = []

    batch_size=256 ## Training batch size
    start_steps=10000 ## Steps sampling random actions
    replay_size=1000000 ## size of replay buffer
    memory = ReplayMemory(seed, replay_size)


    
    for i_episode in range(max_episodes): 
        episode_reward = 0
        episode_steps = 0
        done = False

        qf1_loss = 0
        qf2_loss = 0
        policy_loss = 0
        alpha_loss = 0

        state = env.reset()

        for step in range(max_steps):    
            if start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > batch_size:                
                # Update parameters of all the networks
                qf1_loss, qf2_loss, policy_loss, alpha_loss = agent.update_parameters(memory, batch_size, updates)

                qf1_loss = qf1_loss
                qf2_loss = qf2_loss
                policy_loss = policy_loss
                alpha_loss = alpha_loss

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

        #print("loss:", qf1_loss, qf2_loss, policy_loss, alpha_loss)

        if qf1_loss != 0:
            qf1_loss_array.append(qf1_loss)

        if qf2_loss != 0:
            qf2_loss_array.append(qf2_loss)

        if policy_loss != 0:
            policy_loss_array.append(policy_loss)

        if alpha_loss != 0:
            alpha_loss_array.append(alpha_loss)

        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)        
        avg_score = np.mean(scores_deque)
        std_score = np.std(scores_deque)
        avg_scores_array.append(avg_score)
        std_scores_array.append(std_score)
        max_score = np.max(scores_deque)
        
        if i_episode % 100 == 0 and i_episode > 0:
            reward_round = round(episode_reward, 2)
            save(agent, i_episode, reward_round)
            print('Save environment in episode:  ', i_episode)


        s =  (int)(time.time() - time_start)
            
        print("Ep.: {}, Total Steps: {}, Ep.Steps: {}, Score: {:.3f}, Avg.Score: {:.3f}, Max.Score: {:.3f}, Time: {:02}:{:02}:{:02}".\
            format(i_episode, total_numsteps, episode_steps, episode_reward, avg_score, max_score, \
                  s//3600, s%3600//60, s%60))

                    
        if (avg_score > threshold):
            print('Solved environment with Avg Score:  ', avg_score)
            save(agent, 'final', avg_score)
            break
            
    return np.array(scores_array), np.array(avg_scores_array), np.array(std_scores_array), \
        np.array(qf1_loss_array), np.array(qf2_loss_array), np.array(policy_loss_array), np.array(alpha_loss_array)


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

    scores_array, avg_scores_array, std_scores_array, qf1_loss_array, qf2_loss_array, policy_loss_array, alpha_loss_array = train(
        env=env, agent=Agent(), max_episodes=1000, threshold=2500, max_steps=max_steps, seed=seed)

    #print(qf1_loss_array)
    #print(qf2_loss_array)
    #print(policy_loss_array)
    #print(alpha_loss_array)

    save_score_plot(scores_array, avg_scores_array, std_scores_array, "final_score")

    with open(join(ROOT, 'train_result', 'scores_array.pkl'), 'wb') as f1:
        pickle.dump(scores_array, f1)

    with open(join(ROOT, 'train_result', 'avg_scores_array.pkl'), 'wb') as f2:
        pickle.dump(avg_scores_array, f2)

    save_loss_plot(qf1_loss_array, 'qf1_loss')
    save_loss_plot(qf2_loss_array, 'qf2_loss')
    save_loss_plot(policy_loss_array, 'policy_loss')
    save_loss_plot(alpha_loss_array, 'alpha_loss')

    # ## load
    # with open(join(ROOT, 'train_result', 'scores_array.pkl'), 'rb') as f1:
    #     scores_array = pickle.load(f1)
    # 
    # with open(join(ROOT, 'train_result', 'avg_scores_array.pkl'), 'rb') as f2:
    #     avg_scores_array = pickle.load(f2)
