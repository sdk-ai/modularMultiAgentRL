import gym
import time
import numpy as np
from sims.cartpole import CartPole
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class Simulator(MultiAgentEnv):
    def __init__(self) -> None:
        self.sim = CartPole()
        self.num_agents = 3
        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=(4,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self):
        initial_cart_position = np.random.uniform(-0.1,0.1)
        initial_cart_velocity = 0.0
        initial_pole_angle = np.random.uniform(-0.1,0.1)
        initial_angular_velocity = 0
        self.dones = set()
        config = {
            'cart_mass' : 0.31,
            'pole_mass' : 0.055,
            'pole_length' : 0.4,
            'initial_cart_position' : initial_cart_position,
            'initial_cart_velocity' : initial_cart_velocity,
            'initial_pole_angle' : initial_pole_angle,
            'initial_angular_velocity' :initial_angular_velocity,
            'target_pole_position' : 0.0,
        }

        self.sim.reset(**config)

        for i in range(self.num_agents):
            obs={i:np.array([initial_cart_position,initial_cart_velocity,\
                initial_pole_angle,initial_angular_velocity])}
        print("OBS",obs)
        return obs
    
    def reward_function(self, action_dict):
        reward = -abs(self.sim.state['pole_angle']) 

        return reward

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

            obs.update({i:np.array([self.sim.state['cart_position'],self.sim.state['pole_angle'], \
                self.sim.state['cart_velocity'],self.sim.state['pole_angular_velocity']])})
            rew[i], done[i], info[i] = reward, abs(self.sim.state['pole_angle'])>=1.0, {}
            terminal = terminal and done[i]
        done["__all__"] = terminal
        # print("Rewards",rew)
        # print("Observations",obs)
        # print("TERMINAL",done)

        return obs, rew, done, info

        