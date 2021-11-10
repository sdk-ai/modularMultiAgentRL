import ray
import argparse
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy
from ray.rllib.agents.pg import PGTrainer, PGTFPolicy, PGTorchPolicy
from ray.rllib.agents.sac.sac import SACTrainer
from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy
from ray.rllib.agents.sac.sac_torch_policy import SACTorchPolicy

import os
import shutil
from glob import glob
# Import sim class
from multiagent_cartpole import Simulator
from config import configs

CHECKPOINT_ROOT = "./saved_chkpoints"

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1200)
parser.add_argument("--stop-reward", type=float, default=100.0)
parser.add_argument("--stop-timesteps", type=int, default=1000)
parser.add_argument("--assess", action="store_true")
parser.add_argument("--use-prev-action", action="store_true")
parser.add_argument("--use-prev-reward", action="store_true")
# Users are likely to forget adding command line arguments and clean-restart is destructive, so
# allow users to make a more concious decision
parser.add_argument("--clean-restart", action="store_true") 
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
args = parser.parse_args()

if args.run=="SAC":
# SetUP Algorithms for Policies:
    pTrainer, TFPolicy, TorchPolicy = SACTrainer, SACTFPolicy, SACTorchPolicy
elif args.run=="PPO":
    pTrainer, TFPolicy, TorchPolicy = PPOTrainer, PPOTFPolicy, PPOTorchPolicy



# Training Config:
def load_training_config(args):
    # NOTE Warning: Use this function only after ray is initialized
    # Create Env and Register Sim in ray.tune
    def env_creator(_):
        return Simulator()
    single_env = Simulator()
    register_env("Simulator", env_creator)

    # Env obs, actions and agents
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    num_agents = single_env.num_agents
    
    # Generate policy and assign to agents:
    def gen_policy():
        return (TorchPolicy if args.torch else TFPolicy, obs_space, act_space, {})
    
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()
    
    def policy_mapping_fn(agent_id, **kwargs): # Note removed episode argument, it was unused anyway.
        return 'agent-' + str(agent_id)
    # Def training configs with hyperparam
    # TODO: Add a common training config system to help ease experimentation 
    # Specify Training Configurations:
    if args.assess:
        config = dict(
                configs[args.run],
                **{ "env": "Simulator",  
                    "log_level": "INFO",
                    "num_workers":0,
                    "num_gpus":int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                    "multiagent": {
                        "policies": policy_graphs,
                        "policy_mapping_fn": policy_mapping_fn
                    }
                })
    else:
        config = dict(
                configs[args.run],
                **{ "env": "Simulator",  
                    "log_level": "INFO",
                    "num_workers":2,
                    "num_envs_per_worker": 3,
                    "num_cpus_for_driver": 1,
                    "num_cpus_per_worker": 2,
                    "remote_worker_envs": True,
                    "num_gpus":int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                    "multiagent": {
                        "policies": policy_graphs,
                        "policy_mapping_fn": policy_mapping_fn
                    }
                })
    return config

def clean_start():
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
    ray_results = os.getenv("HOME") + "/ray_results/"
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    #  "checkpoint_"+str(max([int_checkpoint(n[11:]) for n in os.listdir("./saved_chkpoints") \
    #     if os.path.isdir("./saved_chkpoints"+"/"+n)]))
def latest_checkpoint():
    ind_list = [int(n[11:]) for n in os.listdir("./saved_chkpoints") \
        if os.path.isdir("./saved_chkpoints"+"/"+n)]
    ckptmax = [n for n in os.listdir("./saved_chkpoints")][ind_list.index(max(ind_list))]
    return ckptmax, max(ind_list)

# Driver code 
def setup_and_train():
    # args = parser.parse_args()
    ray.init(ignore_reinit_error=True)
    config= load_training_config(args)
    p_train = pTrainer(env="Simulator",config=config)
    # p_train.restore(CHECKPOINT_ROOT+"/checkpoint_000006/checkpoint-6"+".tune_metadata")
    if os.path.isdir(CHECKPOINT_ROOT):
        if args.clean_restart:
            clean_start()
        else:
            print([n for n in os.listdir("./saved_chkpoints") if os.path.isdir("./saved_chkpoints"+"/"+n)])
            ckpt, maxind = latest_checkpoint()
            p_train.restore(CHECKPOINT_ROOT+"/"+ckpt+"/checkpoint-"+str(maxind))

    for i in range(args.stop_iters):
        print("--- Iteration", i, "---")
        result = p_train.train()
        print(pretty_print(result))
        if i%50==0:
            saved_checkpoint = p_train.save(CHECKPOINT_ROOT)
            print("CHECK POINT SAVED AT:")
            print(saved_checkpoint)
    ray.shutdown()

def test():
    single_env = Simulator()
    ray.init(ignore_reinit_error=True)
    _config= load_training_config(args)
    p_train = pTrainer(env="Simulator",config=_config)
    # p_train.restore(CHECKPOINT_ROOT+"/checkpoint_000006/checkpoint-6"+".tune_metadata")
    if os.path.isdir(CHECKPOINT_ROOT):
        if args.clean_restart:
            raise Exception("Error cannot restart training during test")
        else:
            print([n for n in os.listdir("./saved_chkpoints") if os.path.isdir("./saved_chkpoints"+"/"+n)])
            ckpt, maxind = latest_checkpoint()
            p_train.restore(CHECKPOINT_ROOT+"/"+ckpt+"/checkpoint-"+str(maxind))
    else:
        raise Exception("No trained model checkpoint available")

    # examine the trained policy
    # policy = p_train.get_policy()
    # print(policy.model.base_model.summary())
    
    obs = single_env.reset()
    episode_reward = {i:0 for i in range(single_env.num_agents)}

    for i in range(args.stop_iters):
        print("--- Iteration", i, "---")
        # action = p_train.compute_single_action(obs,policy_id='agent-0')
        # obs, reward, done, info = single_env.step(action)
        # episode_reward += reward

        action = {}
        for agent_id, agent_obs in obs.items():
            policy_id = _config['multiagent']['policy_mapping_fn'](agent_id)
            action[agent_id] = p_train.compute_action(agent_obs, policy_id=policy_id)
        obs, reward, done, info = single_env.step(action)
        for j in range(single_env.num_agents):
            episode_reward[j] += reward[j]
        done = done['__all__']
        print("EPISODE REWARD")
        print(pretty_print(episode_reward))
        print("OBSERVATIONS PER AGENT")
        print(pretty_print(obs))
    ray.shutdown()


if __name__=='__main__':
    
    if args.assess:
        test()
    else:
        setup_and_train()


