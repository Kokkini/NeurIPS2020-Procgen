from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "beta": hp.uniform("beta", 1, 10),
        "vae_loss_coeff": hp.loguniform("vae_loss_coeff", np.log(1e-2), np.log(1)),
  }
current_best_params = [
	{
		"beta": 5,
    "vae_loss_coeff": 0.1
	},
  {
		"beta": 5,
    "vae_loss_coeff": 0.01
	},
  {
		"beta": 5,
    "vae_loss_coeff": 1
	},
  {
		"beta": 3,
    "vae_loss_coeff": 0.1
	},
  {
		"beta": 8,
    "vae_loss_coeff": 0.1
	},
]

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)