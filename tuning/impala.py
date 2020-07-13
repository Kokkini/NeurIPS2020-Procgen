from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
		"entropy_coeff": hp.loguniform("entropy_coeff", np.log(1e-3), np.log(1e-1)),
		"vf_loss_coeff": hp.loguniform("vf_loss_coeff", np.log(0.1), np.log(2.5)),
		"num_sgd_iter": scope.int(hp.quniform("num_sgd_iter", 2, 6, 1)),
        "gamma": hp.uniform("gamma", 0.99, 0.999),
        "grad_clip": hp.loguniform("grad_clip", np.log(0.1), np.log(100)),

    }
current_best_params = [
    {
        "lr": 0.0006,
        "entropy_coeff": 0.0019,
        "vf_loss_coeff": 0.45,
        "num_sgd_iter": 2,
        "gamma": 0.999,
        "grad_clip": 40
    }
]

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)