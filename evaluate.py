"""Evaluates an agent on a Walker2DBullet environment."""
from argparse import ArgumentParser, Namespace
from math import fsum
from pickle import dump
from typing import List, Optional

import gym
from gym import logger
# pybullet_envs must be imported in order to create Walker2DBulletEnv
import pybullet_envs    # noqa: F401  # pylint: disable=unused-import

from ee619.agent import Agent


def parse_args() -> Namespace:
    """Parses arguments for evaluate()"""
    parser = ArgumentParser(
        description='Evaluates an agent on a Walker2DBullet environment.')
    parser.add_argument('-l', dest='label', default=None,
                        help='if unspecified, the mean episodic return will be '
                             'printed to stdout. otherwise, it will be dumped '
                             'to a pickle file of the given path.')
    parser.add_argument('-n', type=int, dest='repeat', default=1,
                        help='number of trials.')
    parser.add_argument('-s', type=int, dest='seed', default=0,
                        help='passed to the environment for determinism.')
    return parser.parse_args()


def evaluate(agent: Agent, label: Optional[str], repeat: int, seed: int):
    """Computes the mean episodic return of the agent.

    Args:
        agent: The agent to evaluate.
        label: If None, the mean episodic return will be printed to stdout.
            Otherwise, it will be dumped to a pickle file of the given name
            under the "data" directory.
        repeat: Number of trials.
        seed: Passed to the environment for determinism.
    """
    logger.set_level(logger.DISABLED)
    env = gym.make('Walker2DBulletEnv-v0')
    agent.load()
    rewards: List[float] = []
    for seed_ in range(seed, seed + repeat):
        env.seed(seed_)
        observation = env.reset()
        done = False
        while not done:
            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
    mean_episode_return = fsum(rewards) / repeat
    if label is None:
        print(mean_episode_return)
    else:
        if not label.endswith('.pkl'):
            label += '.pkl'
        with open(label, 'wb') as file:
            dump(mean_episode_return, file)
    return mean_episode_return


if __name__ == '__main__':
    evaluate(agent=Agent(), **vars(parse_args()))
