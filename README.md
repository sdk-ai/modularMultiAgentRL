## RLLIB REPOSITORY FOR EXPLORE PROJECTS

### ENVIRONMENT SET UP

Install anaconda or miniconda in the linux environment.
Follow the steps given here: https://conda.io/projects/conda/en/latest/user-guide/install/linux.html


Use the `environment.yml` files to create a new virtual environment. Run the below command to install libraries specified in the yml file.
```
conda env create -f environment.yml
```
The repository has two parts: 1) Training Agent , 2) Simulator Bridge. The Training Agent is the driver code that imports a simulator and trains an agent based on config specified and uses RLlib policies, algorithms and Ray tune for registering the simulator and also other facilities such as logging. 
The simulator class is imported from a Simulator Bridge code that allows controlling the simulator by using `reset()` and `step()` functionality.

Run `train.py` file which is for training agents using a custom simulator. Training config is set up to
use `PPO` algorithm and using only CPUs, but this can be modified by making changes in the modifiable part of the code indicated by comments.


#TODO: Simulator and Training Configs are in the examples e.g. config in `train.py` and in config in `multiagent_cartpole.py`.

## SIMULATOR BRIDGE

### TEMPLATE
This repo has a Simulator Bridge template `sim_bridge_template.py`. It won't run as-is, but you can add your custom simulator to `/sims` directory, import it and the sim class to the template, and then modify the template based on the docstring and then use it as integrated bridge. Specific sim integration can be implemented in the `sim_bridge_template.py`

### EXAMPLE

The example simulator bridge is a Multiagent Cartpole, `multiagent_cartpole.py`. This is a working example. Two agents actions are summed to provide a single action to the simulator. The two agents have to learn that the cumulative action of the two agents will affect balancing of the pole.

Notes: The multiagent example here is different than one in RLLib. There are a few reasons for having to do that differently: 
    1) the RLlib  repo uses gym env directly to make a multiagent cartpole from the gym cartpole, so we use a separate cartpole model that allows stepping and resetting of the simulator, and modifying states, actions, their observation space, action space and their respective ranges. 
    2) the RLLib example does not train the same type of policies, as we are keeping models independent while the RLlib sample shares weights. This separation might help in agents training independent models and simplifying experimentation. In later stages, experimenters can explore model weight sharing. 
    3) RLlib is using separate environments and observations and actions, while in this example we are sharing global observations and the environment. Moreover, action of two agents is summed.