import gym
import time
import numpy as np
''' IMPORT SIM FROM THE SIMS FOLDER
# from sims.<add_sim> import <sim-class>'''
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class Simulator(MultiAgentEnv):
    def __init__(self) -> None:
        ''' ADD SIM CLASS HERE
        self.sim = <add-sim> '''
        self.num_agents = 3
        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=(4,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self):
        self.dones = set()
        # Add Sim Configs: Episode Configs and INitial Sim States go here:
        ''' ADD SIM CONFIGS HERE:
        config = {
           "sim_config_key":<sim-config-value> # Based on Sim config range, you can dandomize this
        }'''

        self.sim.reset(**config)
        ''' ADD Sim Configs for each agent:
        for i in range(self.num_agents):
            obs={i:np.array([<initial-sim>])}
        return obs '''

    ''' Customize Reward Function
    def reward_function(self, action_dict):
        reward = -abs(self.sim.state[<sim key>]) # example reward, need reshaping

        return reward
        '''

    def step(self, action_dict):
        rew, done, info ={},{},{}
        obs = {}
        # Catch any AI/RLlib agent bugs with NaNs
        if np.isnan(sum(action_dict.values())[0]):
            raise Exception("Actions from agent has NaN! Investigate or Change policy configs")

        self.sim.step(sum(action_dict.values())[0])

        reward = self.reward_function(action_dict)
        terminal = True


        for i in range(self.num_agents):
            # Catch any Sim Bugs with NaNs
            if np.isnan(sum(self.sim.state.values())):
                raise Exception("Invalid States:NaN encountered. Debug Simulator!")
            '''
            obs.update({i:np.array([self.sim.state[<add-sim-key>]])})
            env_terminal = <write env terminal for agent>
            rew[i], done[i], info[i] = reward, env_terminal, {}'''

            terminal = terminal and done[i]
        done["__all__"] = terminal

        return obs, rew, done, info

        