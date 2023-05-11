# import necessary libraries
import os
import random
import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gym
import tensorflow as tf

from IoTEnvironment import create_environment
from agent import Agent

if __name__ == '__main__':
    # create the environment 'IoTEnv-v0'
    env = create_environment()

    # create the agent
    agent = Agent("Agent")

    # train the agent
    agent.train_agent(env)



