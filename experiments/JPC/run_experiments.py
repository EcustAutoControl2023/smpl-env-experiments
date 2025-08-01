import logging
import structlog

logging.basicConfig(level=logging.DEBUG)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

from argparse import Namespace
import d3rlpy
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.models import QRQFunctionFactory, VectorEncoderFactory
from d3rlpy.preprocessing import (
    MinMaxActionScaler,
    MinMaxObservationScaler,
    MinMaxRewardScaler,
)
from d3rlpy.algos import (
    AWACConfig,
    BCConfig,
    BCQConfig,
    BEARConfig,
    CalQLConfig,
    CQLConfig,
    CRRConfig,
    DDPGConfig,
    DQNConfig,
    IQLConfig,
    NFQConfig,
    PLASConfig,
    PRDCConfig,
    ReBRACConfig,
    SACConfig,
    TD3Config,
    TD3PlusBCConfig,
)
from d3rlpy_patch.algos import SACIFConfig, AWCQLConfig, AWSACConfig
from d3rlpy_patch.algos.experts import StaticRecipeExpert
import numpy
from smpl.envs.pensimenv import NUM_STEPS
from utils import env_creator, get_datasets, get_recipe

env_name = "pensimenv"
normalize = False
dense_reward = True
random_seed = 0
env_config = {
    "env_name": env_name,
    "normalize": False,
    "dense_reward": True,
    "random_seed": 0,
    "standard_reward_style": "yield",
    "initial_state_deviation_ratio": 0.05,
}
# Algorithm mappings for cleaner selection
ALGO_MAPPING = {
    "offline": {
        "cql": CQLConfig,
        "bear": BEARConfig,
        "bc": BCConfig,
        "bcq": BCQConfig,
        "ddpg": DDPGConfig,
        "dqn": DQNConfig,
        "iql": IQLConfig,
        "nfq": NFQConfig,
        "plasm": PLASConfig,
        "prdc": PRDCConfig,
        "sac": SACConfig,
        "td3": TD3Config,
        "td3plusbc": TD3PlusBCConfig,
        "awac": AWACConfig,
        "calql": CalQLConfig,
        "crr": CRRConfig,
        "rebrac": ReBRACConfig,
        "awcql": AWCQLConfig,
    },
    "online": {
        "sac": SACConfig,
        "sacif": SACIFConfig,
        "td3": TD3Config,
        "ddpg": DDPGConfig,
        "awsac": AWSACConfig,
    },
}


def get_pensim_env(opts: dict):
    # Create a local copy of env_config to avoid modifying the global version
    local_env_config = env_config.copy()
    local_env_config.update(opts)

    d3rlpy.seed(local_env_config["random_seed"])
    numpy.random.seed(local_env_config["random_seed"])

    env = env_creator(local_env_config)

    # Create a new seed for eval environment
    eval_env_config = local_env_config.copy()
    eval_env_config["random_seed"] += opts.get("random_seed_range", 10)
    eval_env = env_creator(eval_env_config)

    env.reset()
    eval_env.reset()
    return env, eval_env, opts["random_seed"]


def select_algorithm(algo_name: str, algo_type: str):
    algo_name = algo_name.lower()
    if algo_name not in ALGO_MAPPING[algo_type]:
        raise ValueError(f"Unsupported {algo_type} algorithm: {algo_name}")
    return ALGO_MAPPING[algo_type][algo_name]


def initialize_algo(algo_config, args: Namespace, env, device="cuda:0"):
    # Get algorithm-specific parameters
    algo_params = {}

    # Add basic parameters that apply to all algorithms
    algo_params.update(
        {
            "observation_scaler": MinMaxObservationScaler(),
            "action_scaler": MinMaxActionScaler(),
            "critic_encoder_factory": VectorEncoderFactory(
                use_layer_norm=args.use_layer_norm
            ),
        }
    )
    if args.offline_init_policy:
        algo_params.update(
            {
                "reward_scaler": MinMaxRewardScaler(),
            }
        )
    else:
        if args.online_algo == "cql":
            pass
        else:
            algo_params.update(
                {
                    "intervention_method": args.intervention_method,
                    "intervention_rate": args.intervention_rate,
                    "intervention_length": args.intervention_length,
                    "intervention_degree_start": args.intervention_degree_start,
                    "intervention_degree_end": args.intervention_degree_end,
                    "intervention_k": args.intervention_k,
                    "intervention_T": args.intervention_T,
                    # "mixing_ratio": args.mixing_ratio,
                    "buffer_method": args.buffer_method,
                    "buffer_start": args.buffer_start,
                    "buffer_end": args.buffer_end,
                    "buffer_k": args.buffer_k,
                }
            )

    # We don't pass algorithm-specific parameters here to avoid conflicts
    # with the algorithm's default parameters

    return algo_config(**algo_params).create(device=device)


@dataclass
class EnvironmentConfig:
    name: str = "pensimenv"
    dense_reward: bool = True
    normalize: bool = False
    standard_reward_style: str = "yield"
    initial_state_deviation_ratio: float = 0.05


@dataclass
class ModelConfig:
    use_layer_norm: bool = True
    offline_algo: str = "cql"
    online_algo: str = "sacif"
    mixing_ratio: float = 0.0


@dataclass
class BufferConfig:
    method: str = "Constant"  # Options: Constant, Exponential, Linear, Decayless
    start: float = 0.5
    end: float = 0.0
    k: int = 3


@dataclass
class ExplorationConfig:
    use_explorer: bool = False
    explorer_epsilon: float = 0.3


@dataclass
class InterventionConfig:
    method: str = "Linear"  # Options: Linear, Constant, Exponential, ConfidentBased
    rate: float = 0.0
    length: int = 3
    degree_start: float = 1.0  # Start degree for linear intervention
    degree_end: float = 0.0
    k: int = 5  # For linear intervention, number of steps to reach end degree
    T: int = (
        1  # Total number of steps in the environment, used for intervention scheduling
    )


@dataclass
class PolicyInitConfig:
    offline_exp_suffix: str = "for_online"
    offline_init_policy: bool = False
    use_offline_pretrained_model: bool = False
    offline_pretrain_model_path: str = (
        "./d3rlpy_logs/CQL_Pensim_for_online_0_20250318212631/model_1150.d3"
    )
    offline_dataset_path: Optional[str] = None  # Path to offline dataset, if any


@dataclass
class TrainingConfig:
    offline_pretrain_steps: int = 1150
    offline_pretrain_epoch: int = 1
    online_steps: int = 1150  # Same as offline_pretrain_steps by default
    online_epoch: int = 100  # Same as exp_length by default


@dataclass
class OutputConfig:
    save_interval: int = 1


@dataclass
class ExperimentConfig:
    # Experiment configuration
    exp_name: str = "default_exp"
    exp_length: int = 100  # Number of steps for online learning
    random_seed_range: int = 10  # Number of seeds to run
    random_seed: int = 0  # Will be set during experiment run
    experiment: Optional[str] = None  # Name of the experiment to run

    # Nested configurations
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    policy_init: PolicyInitConfig = field(default_factory=PolicyInitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Algorithm-specific configs
    cql: Dict[str, Any] = field(default_factory=dict)
    sacif: Dict[str, Any] = field(default_factory=dict)


def config_to_args(cfg: DictConfig) -> Namespace:
    """Convert a Hydra config to an argparse Namespace.

    Args:
        config: Hydra configuration object

    Returns:
        Namespace object with configuration parameters as attributes
    """
    args = Namespace()

    # Extract configuration from nested structure
    args.env_name = cfg.environment.name
    args.exp_name = cfg.exp_name
    args.exp_length = cfg.exp_length
    args.dense_reward = cfg.environment.dense_reward
    args.random_seed = 0 if "random_seed" not in cfg else cfg.random_seed
    args.random_seed_range = cfg.random_seed_range
    args.use_layer_norm = cfg.model.use_layer_norm
    args.offline_algo = cfg.model.offline_algo
    args.online_algo = cfg.model.online_algo
    # FIXME: Mixing ratio is not used in the current code, but kept for compatibility
    # args.mixing_ratio = cfg.model.mixing_ratio
    args.buffer_method = cfg.buffer.method
    args.buffer_start = cfg.buffer.start
    args.buffer_end = cfg.buffer.end
    args.buffer_k = cfg.buffer.k

    args.use_explorer = cfg.exploration.use_explorer
    args.explorer_epsilon = cfg.exploration.explorer_epsilon

    args.intervention_rate = cfg.intervention.rate
    args.intervention_length = cfg.intervention.length
    args.intervention_method = cfg.intervention.method
    args.intervention_degree_start = cfg.intervention.degree_start
    args.intervention_degree_end = cfg.intervention.degree_end
    args.intervention_k = cfg.intervention.k
    args.intervention_T = cfg.intervention.T

    args.offline_exp_suffix = cfg.policy_init.offline_exp_suffix
    args.offline_init_policy = cfg.policy_init.offline_init_policy
    args.use_offline_pretrained_model = cfg.policy_init.use_offline_pretrained_model
    args.offline_pretrain_model_path = cfg.policy_init.offline_pretrain_model_path
    args.offline_dataset_path = cfg.policy_init.offline_dataset_path
    args.offline_pretrain_steps = cfg.training.offline_pretrain_steps
    args.offline_pretrain_epoch = cfg.training.offline_pretrain_epoch
    args.online_steps = cfg.training.online_steps
    args.online_epoch = cfg.training.online_epoch
    args.save_interval = cfg.output.save_interval

    return args


def run_experiments(args: Namespace):
    env, eval_env, seed = get_pensim_env(
        {
            "env_name": args.env_name,
            "random_seed": args.random_seed,
            "dense_reward": args.dense_reward,
            "random_seed_range": args.random_seed_range,
        }
    )
    recipe_dict = get_recipe(args.env_name)
    explorer = (
        d3rlpy.algos.ConstantEpsilonGreedy(epsilon=args.explorer_epsilon)
        if args.use_explorer
        else None
    )

    # Offline Algorithm Initialization
    offline_algo = None
    if args.offline_init_policy:
        offline_config = select_algorithm(args.offline_algo, "offline")
        if args.use_offline_pretrained_model:
            offline_algo = d3rlpy.load_learnable(args.offline_pretrain_model_path)
        else:
            dataset = get_datasets(args.env_name, args.offline_dataset_path)
            assert dataset is not None
            offline_algo = initialize_algo(offline_config, args, env)
            offline_algo.fit(
                dataset,
                n_steps=args.offline_pretrain_steps * args.offline_pretrain_epoch,
                n_steps_per_epoch=args.offline_pretrain_steps,
                save_interval=1,
                evaluators={
                    "environment": d3rlpy.metrics.EnvironmentEvaluator(env, n_trials=10)
                },
                experiment_name=f"{args.offline_algo}_{args.env_name}_{args.offline_exp_suffix}_{seed}",
            )

    # Online Algorithm Initialization
    online_config = select_algorithm(args.online_algo, "online")
    if args.offline_init_policy:
        assert offline_algo is not None
        online_params = {}

        # Prepare parameters for online algorithm
        if args.online_algo == "cql":
            online_params = {
                "observation_scaler": offline_algo.observation_scaler,
                "action_scaler": offline_algo.action_scaler,
                "reward_scaler": offline_algo.reward_scaler,
                "critic_encoder_factory": VectorEncoderFactory(
                    use_layer_norm=args.use_layer_norm
                ),
            }
        elif args.online_algo == "sacif":
            online_params = {
                "observation_scaler": offline_algo.observation_scaler,
                "action_scaler": offline_algo.action_scaler,
                "reward_scaler": offline_algo.reward_scaler,
                "intervention_method": args.intervention_method,
                "intervention_rate": args.intervention_rate,
                "intervention_length": args.intervention_length,
                "intervention_degree_start": args.intervention_degree_start,
                "intervention_degree_end": args.intervention_degree_end,
                "intervention_k": args.intervention_k,
                "intervention_T": args.intervention_T,
                # "mixing_ratio": args.mixing_ratio,
                "buffer_method": args.buffer_method,
                "buffer_start": args.buffer_start,
                "buffer_end": args.buffer_end,
                "buffer_k": args.buffer_k,
                "critic_encoder_factory": VectorEncoderFactory(
                    use_layer_norm=args.use_layer_norm
                ),
            }

        online_algo = online_config(**online_params).create(device="cuda:0")
        online_algo.build_with_env(env)
        online_algo.copy_policy_from(offline_algo)
        online_algo.copy_q_function_from(offline_algo)
    else:
        online_algo = initialize_algo(online_config, args, env)
        online_algo.build_with_env(env)

    # Online Training
    dataset = get_datasets(args.env_name, args.offline_dataset_path)
    assert dataset is not None
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=env)

    if args.online_algo == "sacif":
        online_algo.fit_online(
            env,
            buffer,
            explorer,
            n_steps=args.online_steps * args.exp_length,
            eval_env=eval_env,
            n_steps_per_epoch=args.online_steps,
            update_start_step=0,
            expert=StaticRecipeExpert(recipe_dict=recipe_dict)
            if recipe_dict is not None
            else None,
            experiment_name=f"{args.online_algo}_env-{args.env_name}_reward-{'dense' if args.dense_reward else 'sparse'}_type-{'offline_guide_online' if args.offline_init_policy else 'online_only'}_explorer-{'on' if args.use_explorer else 'off'}_critic-norm-{'on' if args.use_layer_norm else 'off'}_{args.exp_name}_seed-{seed}",
            offline_buffer=dataset,
            save_interval=args.save_interval,
        )
    else:
        online_algo.fit_online(
            env,
            buffer,
            explorer,
            n_steps=args.online_steps * args.exp_length,
            eval_env=eval_env,
            n_steps_per_epoch=args.online_steps,
            update_start_step=0,
            experiment_name=f"{args.online_algo}_env-{args.env_name}_reward-{'dense' if args.dense_reward else 'sparse'}_type-{'offline_guide_online' if args.offline_init_policy else 'online_only'}_explorer-{'on' if args.use_explorer else 'off'}_critic-norm-{'on' if args.use_layer_norm else 'off'}_{args.exp_name}_seed-{seed}",
            save_interval=args.save_interval,
        )


# Register the config classes with Hydra's config store
cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)
cs.store(name="env_config", node=EnvironmentConfig)
cs.store(name="model_config", node=ModelConfig)
cs.store(name="exploration_config", node=ExplorationConfig)
cs.store(name="intervention_config", node=InterventionConfig)
cs.store(name="policy_init_config", node=PolicyInitConfig)
cs.store(name="training_config", node=TrainingConfig)
cs.store(name="output_config", node=OutputConfig)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for running experiments with Hydra configuration."""
    print("Using Hydra configuration:")
    cfg = cfg.experiment
    print(OmegaConf.to_yaml(cfg))

    # Set initial random_seed if not present
    if "random_seed" not in cfg:
        OmegaConf.update(cfg, "random_seed", 0)

    # Convert Hydra config to argparse Namespace for compatibility
    args = config_to_args(cfg)

    # Run experiments for each seed
    seed_range = args.random_seed_range
    print(f"Running {seed_range} seeds...")
    for seed in range(cfg.random_seed, seed_range):
        args.random_seed = seed
        print(f"\n=== Running experiment with seed {seed} ===\n")
        run_experiments(args)

    print(f"\n=== All experiments completed successfully ===\n")

    return


if __name__ == "__main__":
    # python run_new_experiments.py experiment=rate0_cql_sacif
    # python run_new_experiments.py experiment=simple_test
    # python run_new_experiments.py experiment=rate0_cql_sacif model.use_layer_norm=false
    try:
        # Initialize Hydra with empty config
        main()
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback

        traceback.print_exc()
