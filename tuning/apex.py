from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
        "target_network_update_freq": scope.int(hp.loguniform("target_network_update_freq", np.log(1e4), np.log(1e5))),
        "n_step": scope.int(hp.quniform("n_step", 3, 12, 1)),
        "num_atoms": scope.int(hp.quniform("num_atoms", 20, 100, 1))
    }
current_best_params = [
    {
        "lr": 1e-4,
        "target_network_update_freq": 32000,
        "num_atoms": 51,
        "n_step": 3
    }
]

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)