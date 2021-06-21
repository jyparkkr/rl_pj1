"""Train an agent on a Walker2DBullet environment."""
from os.path import abspath, dirname, realpath, join
from collections import deque

import gym
from gym import logger
# pybullet_envs must be imported in order to create Walker2DBulletEnv
import pybullet_envs    # noqa: F401  # pylint: disable=unused-import
import numpy as np
import torch
from ee619.agent import Agent
from ee619.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import pickle

import time


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory


def save(agent, episode, reward):
    """Save state of agent.

    Args:
        agent: The agent on train.
        episode: Number of episode on current training step.
        reward: Reward on current training step.
    """
    policy_path = f"weights_policy_ep_{episode:>05}_rw_{reward:5.2f}.pth"
    critic_q1_path = f"weights_critic_q{1}_ep_{episode:>05}_rw_{reward:5.2f}.pth"
    critic_q2_path = f"weights_critic_q{2}_ep_{episode:>05}_rw_{reward:5.2f}.pth"

    policy = join(ROOT, 'saved_model', policy_path)
    critic_q1 = join(ROOT, 'saved_model', critic_q1_path)
    critic_q2 = join(ROOT, 'saved_model', critic_q2_path)

    if episode == 'final':
        policy = join(ROOT, 'saved_model', 'weights_policy_final.pth')
        critic_q1 = join(ROOT, 'saved_model', 'weights_critic_q1_final.pth')
        critic_q2 = join(ROOT, 'saved_model', 'weights_critic_q2_final.pth')


    torch.save(agent.policy.state_dict(), policy)
    torch.save(agent.critic_q1.state_dict(), critic_q1)
    torch.save(agent.critic_q2.state_dict(), critic_q2)


def save_score_plot(ep_score, avg_score, std_score, score):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(left = 0, right = len(ep_score))
    plt.plot(np.arange(1, len(ep_score)+1), ep_score, label="Score")
    plt.plot(np.arange(1, len(avg_score)+1), avg_score, label="Avg on 100 episodes")
    plt.fill_between(
        np.arange(1, len(avg_score)+1), 
        avg_score - std_score, 
        avg_score + std_score, 
        color='gray', 
        alpha=0.2
    )

    plt.legend() 
    plt.ylabel('Score')
    plt.xlabel('Episodes #')
    if score is not None:
        path = join(ROOT, 'train_result', score + '_score' + '.png')
        plt.savefig(path)

    plt.clf()


def save_loss_plot(start_ep, losses, name, score):
    #import matplotlib.pyplot as plt

    #print('length of losses: ', len(losses))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(left = 0, right = start_ep + len(losses))
    plt.plot(np.arange(start_ep, start_ep + len(losses)), losses, label= name)

    plt.legend() 
    plt.ylabel('Loss')
    plt.xlabel('Episodes #')
    if score is not None:
        path = join(ROOT, 'train_result', score + '_' +name + '.png')
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
    loss_start_ep = 0

    import time
    time_start = time.time()
    scores_deque = deque(maxlen=100)
    ep_score_array = []
    avg_score_array = [] 
    std_score_array = []

    qf1_loss_array = []
    qf2_loss_array = []
    policy_loss_array = []
    alpha_loss_array = []

    # Hyperparameter sizes are on Appendix D
    minibatch_size = 256 ## Training batch size
    start_steps = 10000 ## Steps sampling random actions
    replay_size = 1000000 ## size of replay buffer
    buffer = ReplayBuffer(seed, replay_size)


    for i_ep in range(max_episodes): 
        ep_reward = 0
        ep_steps = 0
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

            if len(buffer) > minibatch_size:                
                # Update parameters of all the networks
                qf1_loss, qf2_loss, policy_loss, alpha_loss = agent.update_parameters(buffer, minibatch_size)

                qf1_loss = qf1_loss
                qf2_loss = qf2_loss
                policy_loss = policy_loss
                alpha_loss = alpha_loss

                updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            ep_steps += 1
            total_numsteps += 1
            ep_reward += reward

            mask = 1 if ep_steps == env._max_episode_steps else float(not done)

            buffer.push(state, action, reward, next_state, mask) # Append transition to buffer

            state = next_state
            
            if done:
                break

        if updates > 0:
            qf1_loss_array.append(qf1_loss)
            qf2_loss_array.append(qf2_loss)
            policy_loss_array.append(policy_loss)
            alpha_loss_array.append(alpha_loss)

        else:
            loss_start_ep += 1

        scores_deque.append(ep_reward)
        ep_score_array.append(ep_reward)        
        avg_score = np.mean(scores_deque)
        std_score = np.std(scores_deque)
        avg_score_array.append(avg_score)
        std_score_array.append(std_score)
        max_score = np.max(scores_deque)
        
        if i_ep % 100 == 0 and i_ep > 0:
            save(agent, i_ep, avg_score)
            print('Save environment in episode: ', i_ep)

        import time
        s =  (int)(time.time() - time_start)
        time = f"{s//3600:02}:{s%3600//60:02}:{s%60:02}"
            
        print(f"Ep.: {i_ep}, Ep.Steps: {ep_steps}, Score: {ep_reward:.3f}, Avg.Score: {avg_score:.2f}, Max.Score: {max_score:.2f}, Time: {time}")


        if (avg_score > threshold) or (i_ep == max_episodes - 1):
            print('Solved environment with Avg Score: ', avg_score)
            save(agent, 'final', avg_score)
            break
    
    result = ["ep_score", "avg_score", "std_score", "qf1_loss", "qf2_loss", "policy_loss", "alpha_loss"]
    result_val = [ep_score_array, avg_score_array, std_score_array, qf1_loss_array, qf2_loss_array, policy_loss_array, alpha_loss_array]
    result_val = [np.array(x) for x in result_val]
    train_result = dict(zip(result, result_val))
    train_result["loss_start_ep"] = loss_start_ep
    return train_result


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

    train_result = train(env=env, agent=Agent(), max_episodes=20000, threshold=2500, max_steps=max_steps, seed=seed)

    score = str(train_result['avg_score'][-1])[:7]

    save_score_plot(train_result['ep_score'], train_result['avg_score'], train_result['std_score'], score)
    for result in ["ep_score", "avg_score", "std_score"]:
        with open(join(ROOT, 'train_result', score + f'_{result}.pkl'), 'wb') as f:
            pickle.dump(train_result[result], f)

    for result in ["qf1_loss", "qf2_loss", "policy_loss", "alpha_loss"]:
        save_loss_plot(train_result['loss_start_ep'], train_result[result], result, score)
        with open(join(ROOT, 'train_result', score + f'_{result}.pkl'), 'wb') as f:
            pickle.dump(train_result[result], f)


    # ## load
    # with open(join(ROOT, 'train_result', score + '_ep_score_array.pkl'), 'rb') as f1:
    #     ep_score_array = pickle.load(f1)
    # 
    # with open(join(ROOT, 'train_result', score + '_avg_score_array.pkl'), 'rb') as f2:
    #     avg_score_array = pickle.load(f2)
