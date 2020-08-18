from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
        "gamma": hp.uniform("gamma", 0.98, 1),
        "entropy_coeff": hp.loguniform("entropy_coeff", np.log(1e-2), np.log(1e-1)),
        "clip_rewards": hp.choice("clip_rewards", [True, False]),
        "vf_clip_param": hp.loguniform("vf_clip_param", np.log(1e-1), np.log(1e1)),
    }
current_best_params = [
    {
        "lr": 0.0006,
        "gamma": 0.9,
        "entropy_coeff": 0.01,
        "clip_rewards": True,
        "vf_clip_param": 0.4,
    }
]

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)