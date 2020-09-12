from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(6e-4)),
        "vf_loss_coeff": hp.loguniform("vf_loss_coeff", np.log(5e-2), np.log(1)),
        "sgd_minibatch_size": hp.choice("sgd_minibatch_size", [256, 512])
  }
current_best_params = [
	{
		"lr": 0.0001,
    "vf_loss_coeff": 0.5,
    "sgd_minibatch_size": 1,
	}
]

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)