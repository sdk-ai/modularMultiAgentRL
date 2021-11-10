## RLLIB REPOSITORY FOR EXPLORE PROJECTS

### ENVIRONMENT SET UP

Install anaconda or miniconda in the linux environment.
Follow the steps given here: https://conda.io/projects/conda/en/latest/user-guide/install/linux.html


Use the `environment.yml` files to create a new virtual environment. Run the below command to install libraries specified in the yml file.
```bash
conda env create -f environment.yml
conda activate rllb
```

### Repository Structure
The repository has two parts: 1) Training Agent , 2) Simulator Bridge. The Training Agent is the driver code that imports a simulator,trains an agent based on config specified, and uses RLlib policies, algorithms and/or Ray tune for registering the simulator and also other facilities such as logging. This is done using `train.py`. The simulator class is imported from a Simulator Bridge code that allows controlling the simulator by using `reset()` and `step()` functionality. A simulator bridge template is provided, please see `simulator_bridge_template.py` for details. A complete example that runs with `train.py` is also included: `multiagent_cartpole.py`, and it allows training single or multiple agents.

#### File Structure
```bash
├───.gitignore
├───README.md
├───environment.yml
├───multiagent_cartpole.py
├───sim_bridge_template.py
├───train.py
├───sims
│   ├───cartpole.py
│   ├───render.py
```
 
Steps to start training:
    1. Add the simulator to sims directory or optionally integrate directly into a class in `simulator_bridge_template.py`.
    2. Modify the outlined sim specifics in the `simulator_bridge_template.py`. Change simulator configs that usually include iniital states.
    3. Run `train.py` file which is for training agents using a custom simulator as below:
        ```bash
            python train.py
        ```
    Training config can be added either in the command line or edited in the `train.py`. Default set up uses `PPO` algorithm and only CPUs, but this can be modified by making changes in the modifiable part of the code indicated by comments.
    (Note: The configs design will be modified in near future.)

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

Note: `render.py` is currently not used. This can be added later based on when Linux VM or Desktop requirements are completed.

### LOGGING AND VISUALIZATIONS

For now, we will use tensorboard for visualizing training progress. All logs of experiment configs and training progress are saved  at `~/ray_results`. To visualize progress, open a terminal at `$HOME` and
run below:

```bash
tensorboard --logdir=~/ray_results
```


### Training and Experimentation Settings:

#### Termination criteria
You can change the termination criteria by changing the stop criteria either by modifying in lines 22-24, or by passing using command line. For example, to increase number of iterations, raise `stop-iters`, you can modify
```python
parser.add_argument("--stop-iters", type=int, default=10000)
parser.add_argument("--stop-reward", type=float, default=100.0)
parser.add_argument("--stop-timesteps", type=int, default=10000)
```

### INFERENCING FROM A TRAINED MODEL:

The trained model can be evaluated using `--assess` while running the `train.py` script. This will use the last checkpoint, and run an inference. The roll out script from RLlib requires supplying `env` separately, and would be less convenient.