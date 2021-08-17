import ray
import argparse
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.pg import PGTrainer, PGTFPolicy, PGTorchPolicy
from ray.rllib.agents.registry import get_trainer_class
import os
# Import sim class
from sim import IrrigationEnv

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=150)
parser.add_argument("--stop-reward", type=float, default=1000.0)
parser.add_argument("--stop-timesteps", type=int, default=100000)

# Driver code 
def setup_and_train():
    args = parser.parse_args()
    ray.init()

    def env_creator(_):
        return IrrigationEnv()
    single_env = IrrigationEnv()
    register_env("IrrigationEnv", env_creator)
    
    trainer = "PG"

    # Env obs, actions and agents
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    num_agents = single_env.num_agents

    def gen_policy():
        return (PGTorchPolicy if args.torch else PGTFPolicy, obs_space, act_space, {})
    
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()
    print("Policy Graphs")
    print(policy_graphs)
    
    def policy_mapping_fn(agent_id):
        return 'agent-' + str(agent_id)

    # Def training configs with hyperparam
    # config={    
    #             "log_level": "DEBUG",
    #             "num_workers":2,
    #             "num_cpus_for_driver": 1,
    #             "num_cpus_per_worker": 1,
    #             "num_gpus":int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    #             "simple_optimizer": True,
    #             "num_sgd_iter": 10,
    #             "train_batch_size": 128,
    #             "lr": 5e-3,
    #             "model":{"fcnet_hiddens": [8,8]},
    #             "multiagent": {
    #                 "policies": policy_graphs,
    #                 "policy_mapping_fn": policy_mapping_fn
    #             }
    #     }
    config={
            "log_level": "WARN",
            "num_workers": 3,
            "num_cpus_for_driver": 1,
            "num_cpus_per_worker": 1,
            "lr": 5e-3,
            "model":{"fcnet_hiddens": [8, 8]},
            "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
            },
            "env": "IrrigationEnv"
        }

    pg_train = PGTrainer(env="IrrigationEnv",config=config)
    for i in range(args.stop_iters):
        print("== Iteration", i, "==")
        result_pg = pg_train.train()
        print(pretty_print(result_pg))

if __name__=='__main__':
    args = parser.parse_args()
    setup_and_train()


