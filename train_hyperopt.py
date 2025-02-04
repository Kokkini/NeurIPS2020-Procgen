#!/usr/bin/env python

import argparse
import os
from pathlib import Path
import yaml
import numpy as np

import ray
from ray.cluster_utils import Cluster
from ray.tune.config_parser import make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler, run_experiments
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from utils.loader import load_envs, load_models, load_algorithms
from callbacks import CustomCallbacks

# Try to import both backends for flag checking/warnings.
tf = try_import_tf()
torch, _ = try_import_torch()

"""
Note : This script has been adapted from :
    https://github.com/ray-project/ray/blob/master/rllib/train.py
"""

EXAMPLE_USAGE = """
Training example:
    python ./train.py --run DQN --env CartPole-v0

Training with Config:
    python ./train.py -f experiments/simple-corridor-0.yaml


Note that -f overrides all other trial-specific command-line options.
"""

# Register all necessary assets in tune registries
load_envs(os.getcwd()) # Load envs
load_models(os.getcwd()) # Load models
# Load custom algorithms
from algorithms import CUSTOM_ALGORITHMS
load_algorithms(CUSTOM_ALGORITHMS)

print(ray.rllib.contrib.registry.CONTRIBUTED_ALGORITHMS)

def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--ray-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
        "of starting a new one.")
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-nodes",
        default=None,
        type=int,
        help="Emulate multiple cluster nodes for debugging.")
    parser.add_argument(
        "--ray-redis-max-memory",
        default=None,
        type=int,
        help="--redis-max-memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-memory",
        default=None,
        type=int,
        help="--memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-object-store-memory",
        default=None,
        type=int,
        help="--object-store-memory to use if starting a new cluster.")
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_RESULTS_DIR,
        type=str,
        help="Local dir to save training results to. Defaults to '{}'.".format(
            DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--upload-dir",
        default="",
        type=str,
        help="Optional URI to sync training results to (e.g. s3://bucket).")
    parser.add_argument(
        "-v", action="store_true", help="Whether to use INFO level logging.")
    parser.add_argument(
        "-vv", action="store_true", help="Whether to use DEBUG level logging.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume previous Tune experiments.")
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Whether to use PyTorch (instead of tf) as the DL framework.")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Whether to attempt to enable TF eager execution.")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Whether to attempt to enable tracing for eager mode.")
    parser.add_argument(
        "--env", default=None, type=str, help="The gym environment to use.")
    parser.add_argument(
        "--queue-trials",
        action="store_true",
        help=(
            "Whether to queue trials when the cluster does not currently have "
            "enough resources to launch one. This should be set to True when "
            "running on an autoscaling cluster to enable automatic scale-up."))
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    parser.add_argument(
        "--vf-loss-coeff",
        type=float,
        default=None)
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int)
    parser.add_argument(
        "--timesteps-total",
        default=None,
        type=int)
    parser.add_argument(
        "--lr",
        default=None,
        type=float)
    parser.add_argument(
        "--entropy-coeff",
        default=None,
        type=float)
    parser.add_argument(
        "--vf-clip-param",
        default=None,
        type=float)
    parser.add_argument(
        "--batch-mode",
        default=None,
        type=str)
    parser.add_argument(
        "--num-sgd-iter",
        default=None,
        type=int)
    parser.add_argument(
        "--exploration-config",
        default=None,
        type=str)
    parser.add_argument(
        "--double-q",
        default=None,
        type=str)
    parser.add_argument(
        "--dueling",
        default=None,
        type=str)
    parser.add_argument(
        "--noisy",
        default=None,
        type=str)
    parser.add_argument(
        "--grayscale",
        default=None,
        type=str)
    parser.add_argument(
        "--num-cpus-per-worker",
        default=None,
        type=float)
    parser.add_argument(
        "--num-gpus-per-worker",
        default=None,
        type=float)
    parser.add_argument(
        "--num-gpus",
        default=None,
        type=float)
    parser.add_argument(
        "--buffer-size",
        default=None,
        type=int)
    parser.add_argument(
        "--num-envs-per-worker",
        default=None,
        type=int)
    parser.add_argument(
        "--target-network-update-freq",
        default=None,
        type=int)
    parser.add_argument(
        "--n-step",
        default=None,
        type=int)
    parser.add_argument(
        "--num-atoms",
        default=None,
        type=int)
    parser.add_argument(
        "--train-batch-size",
        default=None,
        type=int)
    parser.add_argument(
        "--tuning-file",
        default=None,
        type=str)
    return parser


def run(args, parser):
    args_dict = vars(args)
    for key in args_dict:
        if args_dict[key] in ["True", "False"]:
            args_dict[key] = args_dict[key]=="True"
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)
            print(experiments)
            for exp in experiments.values():
                if args_dict["timesteps_total"] is not None:
                    exp["stop"]["timesteps_total"] = args_dict["timesteps_total"]
                if args_dict["exploration_config"] is not None:
                    exp["config"]["exploration_config"]["type"] = args_dict["exploration_config"]
                if args_dict["grayscale"] is not None:
                    exp["config"]["model"]["grayscale"] = args_dict["grayscale"]
                for a in args_dict:
                    if a in exp["config"] and args_dict[a] is not None: exp["config"][a] = args_dict[a]
    else:
        # Note: keep this in sync with tune/config_parser.py
        experiments = {
            args.experiment_name: {  # i.e. log to ~/ray_results/default
                "run": args.run,
                "checkpoint_freq": args.checkpoint_freq,
                "keep_checkpoints_num": args.keep_checkpoints_num,
                "checkpoint_score_attr": args.checkpoint_score_attr,
                "local_dir": args.local_dir,
                "resources_per_trial": (
                    args.resources_per_trial and
                    resources_to_json(args.resources_per_trial)),
                "stop": args.stop,
                "config": dict(args.config, env=args.env),
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }
    
    verbose = 1
    for exp in experiments.values():
        # Bazel makes it hard to find files specified in `args` (and `data`).
        # Look for them here.
        # NOTE: Some of our yaml files don't have a `config` section.
        if exp.get("config", {}).get("input") and \
                not os.path.exists(exp["config"]["input"]):
            # This script runs in the ray/rllib dir.
            rllib_dir = Path(__file__).parent
            input_file = rllib_dir.absolute().joinpath(exp["config"]["input"])
            exp["config"]["input"] = str(input_file)

        if not exp.get("run"):
            parser.error("the following arguments are required: --run")
        if not exp.get("env") and not exp.get("config", {}).get("env"):
            parser.error("the following arguments are required: --env")
        if args.eager:
            exp["config"]["eager"] = True
        if args.torch:
            exp["config"]["use_pytorch"] = True
        if args.v:
            exp["config"]["log_level"] = "INFO"
            verbose = 2
        if args.vv:
            exp["config"]["log_level"] = "DEBUG"
            verbose = 3
        if args.trace:
            if not exp["config"].get("eager"):
                raise ValueError("Must enable --eager to enable tracing.")
            exp["config"]["eager_tracing"] = True

        ### Add Custom Callbacks
        exp["config"]["callbacks"] = CustomCallbacks

    if args.ray_num_nodes:
        cluster = Cluster()
        for _ in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
                memory=args.ray_memory,
                redis_max_memory=args.ray_redis_max_memory)
        ray.init(address=cluster.address)
    else:
        ray.init(
            address=args.ray_address,
            object_store_memory=args.ray_object_store_memory,
            memory=args.ray_memory,
            redis_max_memory=args.ray_redis_max_memory,
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus)
    
    from hyperopt import hp
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from hyperopt.pyll.base import scope
    import importlib

    tuning_module = importlib.import_module(f"tuning.{args.tuning_file}")
    algo = tuning_module.algo

    for exp in experiments.values():
        exp["num_samples"] = args.num_samples
        exp["local_dir"] = args.local_dir
    print(f"default result dir: {DEFAULT_RESULTS_DIR}")
    print(f"result dir: {args.local_dir}")
    print(experiments)
    run_experiments(
        experiments,
        search_alg=algo,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume,
        verbose=verbose,
        concurrent=True)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
