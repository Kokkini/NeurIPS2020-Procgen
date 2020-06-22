import numpy as np
import argparse
import os
parser = argparse.ArgumentParser()

parser.add_argument(
    "--num-exp",
    default=20,
    type=int)
parser.add_argument(
    "--logdir",
    default="log",
    type=str)
parser.add_argument(
    "--config",
    default="experiments/impala-baseline.yaml",
    type=str)

args = parser.parse_args()
os.makedirs(args.logdir, exist_ok=True)
with open("tune.sh", "w") as f:
    f.write("#!/bin/bash\n")
    for i in range(args.num_exp):
        d = {}
        d["lr"] = np.exp(np.random.uniform(np.log(1e-4), np.log(25e-4)))
        d["vf-loss-coeff"] = np.exp(np.random.uniform(np.log(0.1), np.log(2.5)))
        d["entropy-coeff"] = np.exp(np.random.uniform(np.log(1e-3), np.log(1e-1)))
        d["vf-clip-param"] = np.exp(np.random.uniform(np.log(0.04), np.log(1)))
        d["batch-mode"] = np.random.choice(["truncate_episodes", "complete_episodes"])
        d["num-sgd-iter"] = np.round(np.exp(np.random.uniform(np.log(1), np.log(9)))).astype(np.int)
        command = 'python train_tune.py -f ${EXPERIMENT:-"' + args.config + '"} --timesteps-total 3000000 --ray-memory ${RAY_MEMORY_LIMIT:-1500000000} --ray-num-cpus ${RAY_CPUS:-2} --ray-object-store-memory ${RAY_STORE_MEMORY:-1000000000}'
        for key in d:
            command += f" --{key} {d[key]}"
        logfile = os.path.join(args.logdir, f"exp_{i:03}.txt")
        command += f" >> \"{logfile}\""
        f.write(command+"\n")
        with open(logfile, "a") as f_log:
            f_log.write(command+"\n")



