from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
        "gamma": hp.uniform("gamma", 0.9, 1),

    }
current_best_params = [
    {
        "lr": 0.0006,
        "gamma": 0.9,
    }
]

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)