from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "lr": hp.loguniform("lr", np.log(4e-5), np.log(1e-3)),
        "entropy_coeff": hp.loguniform("entropy_coeff", np.log(3e-3), np.log(1e-1)),
        "num_sgd_iter": hp.choice("num_sgd_iter",[2,3,4,5]),
        "vf_loss_coeff": hp.loguniform("vf_loss_coeff", np.log(1e-2), np.log(1)),
    }
current_best_params = [
	{
		"lr": 0.0005,
        "entropy_coeff": 0.01,
        "num_sgd_iter": 3,
        "vf_loss_coeff": 1.0,
	}
]

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)