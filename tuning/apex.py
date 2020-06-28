from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
        "n_step": scope.int(hp.quniform("num_sgd_iter", 3, 12, 1)),
        "gamma": hp.choice("gamma", [0.99, 0.999]),
    }
current_best_params = [
    {
        "lr": 1e-4,
        "n_step": 3,
        "gamma": 0.99,

    }
]

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)