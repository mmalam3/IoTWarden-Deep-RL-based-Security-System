# include paths for modules to import
import sys
sys.path.append('./iot_environment/')
sys.path.append('./src/')

# import necessary modules
from IoTEnvironment import create_environment
from agent import Agent

if __name__ == '__main__':
    # create the environment 'IoTEnv-v0'
    env = create_environment()

    # create the agent
    agent = Agent('agent-defender')

    # train the agent
    agent.train_agent(env)



