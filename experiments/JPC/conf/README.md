# Hydra Configuration System for Experiments

This directory contains the Hydra configuration files for running reinforcement learning experiments. Hydra provides a powerful and flexible way to manage experiment configurations, allowing for easy composition, overriding, and multirun capabilities.

## Directory Structure

```
conf/
├── config.yaml                 # Base configuration
├── algorithm/                  # Algorithm-specific configurations
│   ├── cql.yaml                # CQL algorithm configuration
│   └── sacif.yaml              # SACIF algorithm configuration
├── environment/                # Environment-specific configurations
│   └── pensimenv.yaml          # PenSim environment configuration
└── experiment/                 # Predefined experiment configurations
    ├── rate0_cql_sacif.yaml    # Main experiment configuration (CQL+SACIF on PenSim)
    └── simple_test.yaml        # Simple test configuration
```

## Getting Started

### Running an Experiment with Default Configuration

```bash
python run_new_experiments.py
```

### Running with a Specific Experiment Configuration

```bash
python run_new_experiments.py experiment=rate0_cql_sacif
```

You can also try a simpler test configuration:

```bash
python run_new_experiments.py experiment=simple_test
```

Each experiment configuration in the `experiment/` directory automatically combines algorithm and environment configurations through the `defaults` section.

### Overriding Configuration Values

You can override any configuration parameter directly from the command line:

```bash
python run_new_experiments.py env.dense_reward=false model.use_layer_norm=true
```

### Running Multiple Experiments (Sweep)

Hydra supports running multiple experiments with different parameters:

```bash
python run_new_experiments.py --multirun exploration.explorer_epsilon=0.1,0.2,0.3
```

### Overriding Configuration Values

You can override any configuration parameter directly from the command line:

```bash
python run_new_experiments.py experiment=rate0_cql_sacif intervention.rate=0.2
```

This lets you modify specific parameters without creating a new configuration file.

## Configuration Structure

The configuration is organized in a modular way to promote reusability and clarity:

### Base Configuration (config.yaml)

The base configuration defines default values for all parameters and references the configuration groups:

```yaml
defaults:
  - _self_
  - optional algorithm: null
  - optional environment: null
  - optional experiment: null
```

It includes default values for:
- Experiment parameters (name, length, seeds)
- Environment settings
- Model configuration
- Exploration parameters
- Intervention settings
- Policy initialization
- Training parameters

### Algorithm Configurations

Each algorithm has its own configuration file with algorithm-specific parameters:

- `algorithm/cql.yaml`: Configuration for CQL (Conservative Q-Learning)
  ```yaml
  # @package _group_
  # Configuration for CQL algorithm
  
  # Algorithm settings for model configuration
  model:
    offline_algo: "cql"
    use_layer_norm: true
  
  # CQL-specific parameters
  cql:
    alpha: 1.0
    conservative_weight: 5.0
    # other parameters...
  ```

- `algorithm/sacif.yaml`: Configuration for SACIF (Soft Actor-Critic with Intervention Feedback)
  ```yaml
  # @package _group_
  # Configuration for SACIF algorithm
  
  # Algorithm settings for model configuration
  model:
    online_algo: "sacif"
    use_layer_norm: true
  
  # SACIF-specific parameters
  sacif:
    alpha: 0.2
    # other parameters...
  ```

### Environment Configurations

Environment-specific configurations:

- `environment/pensimenv.yaml`: Configuration for the PenSim environment
  ```yaml
  # @package _group_
  # Configuration for PenSim environment
  
  # Environment configuration
  env:
    name: "pensimenv"
    dense_reward: true
    # other settings...
  
  # PenSim-specific parameters
  pensimenv:
    batch_size: 100000
    # other parameters...
  ```

### Experiment Configurations

Ready-made configurations that combine algorithms and environments:

- `experiment/rate0_cql_sacif.yaml`: Main experiment configuration
  ```yaml
  # @package _group_
  # Configuration for CQL+SACIF on PenSim environment
  
  # Set up the configuration composition
  defaults:
    - /algorithm/cql        # Use CQL for offline algorithm
    - /algorithm/sacif      # Use SACIF for online algorithm
    - /environment/pensimenv # Use PenSim environment
    - _self_                # Apply this config last
  
  exp_name: "rate0_cql-sacifpd"
  # other experiment settings...
  ```

- `experiment/simple_test.yaml`: Simple test configuration with fewer iterations

## Key Parameters

### Experiment Parameters
- `exp_name`: Experiment name for logging
- `exp_length`: Number of steps for online learning
- `random_seed_range`: Number of seeds to run (each seed becomes a separate experiment)

### Model Configuration
- `model.offline_algo`: Offline algorithm to use (e.g., "cql")
- `model.online_algo`: Online algorithm to use (e.g., "sacif")
- `model.use_layer_norm`: Whether to use layer normalization

### Training Parameters
- `training.offline_pretrain_steps`: Steps per epoch for offline pretraining
- `training.offline_pretrain_epoch`: Number of epochs for offline pretraining
- `training.online_steps`: Steps per epoch for online learning
- `training.online_epoch`: Number of epochs for online learning

### Policy Initialization
- `policy_init.offline_init_policy`: Whether to use offline RL for initial policy
- `policy_init.use_offline_pretrained_model`: Whether to use a pretrained model
- `policy_init.offline_pretrain_model_path`: Path to the pretrained model

### Exploration Settings
- `exploration.use_explorer`: Whether to use exploration
- `exploration.explorer_epsilon`: Epsilon value for exploration

### Intervention Settings
- `intervention.rate`: Expert intervention rate
- `intervention.length`: Expert intervention length

## Creating New Configurations

### Creating a New Algorithm Configuration

1. Create a new YAML file in the `algorithm/` directory
2. Add algorithm-specific parameters
3. Override any base parameters as needed
## Creating New Configurations

### Creating a New Experiment Configuration

1. Create a new YAML file in the `experiment/` directory
2. Use `# @package _group_` at the top of the file
3. Include the defaults section to specify which algorithm and environment to use
4. Define your experiment-specific parameters

Example of a complete experiment configuration:

```yaml
# @package _group_
# My experiment description

defaults:
  - /algorithm/cql        # Use CQL for offline algorithm
  - /algorithm/sacif      # Use SACIF for online algorithm
  - /environment/pensimenv # Use PenSim environment
  - _self_                # Apply this config last

exp_name: "my_new_experiment"
exp_length: 50
random_seed_range: 5

# Override specific parameters
model:
  use_layer_norm: true

policy_init:
  offline_init_policy: true
  
# Training parameters
training:
  offline_pretrain_steps: 1000
  offline_pretrain_epoch: 1
  online_steps: 1000
  online_epoch: 50
```

### Creating a New Algorithm Configuration

1. Create a new YAML file in the `algorithm/` directory
2. Use `# @package _group_` at the top of the file
3. Define the algorithm settings and parameters

Example:

```yaml
# @package _group_
# Configuration for new algorithm

# Algorithm settings for model configuration
model:
  offline_algo: "new_algorithm"  # or online_algo depending on the type
  use_layer_norm: true

# Algorithm-specific parameters
new_algorithm:
  learning_rate: 0.001
  gamma: 0.99
  # other parameters...

# Default policy initialization for this algorithm
policy_init:
  offline_init_policy: true
```

### Creating a New Environment Configuration

1. Create a new YAML file in the `environment/` directory
2. Use `# @package _group_` at the top of the file
3. Define the environment settings and parameters

Example:

```yaml
# @package _group_
# Configuration for new environment

# Environment configuration
env:
  name: "new_environment"
  dense_reward: true
  normalize: false
  # other settings...

# Environment-specific parameters
new_environment:
  max_steps: 1000
  termination_penalty: -100
  # other parameters...

# Default environment-specific training parameters
training:
  steps_per_epoch: 1000
```

## Benefits of Using Hydra

- **Modularity**: Separate algorithm, environment, and experiment configurations
- **Composition**: Easily combine algorithms and environments through defaults lists
- **Overrides**: Change parameters without modifying config files
- **Multirun**: Run the same code with different configurations
- **Logging**: Automatic logging of all configuration parameters
- **Searchable outputs**: Each run is stored in a timestamped directory

## Running Experiments

### Basic Usage

```bash
# Run with a specific experiment configuration
python run_new_experiments.py experiment=rate0_cql_sacif

# Override specific parameters
python run_new_experiments.py experiment=rate0_cql_sacif intervention.rate=0.2

# Run multiple experiments with different parameters
python run_new_experiments.py experiment=rate0_cql_sacif --multirun intervention.rate=0.0,0.1,0.2
```

For more information on Hydra, see the [official documentation](https://hydra.cc/docs/intro).