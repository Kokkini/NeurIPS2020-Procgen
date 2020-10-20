from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "fast_loss_coeff": hp.uniform("fast_loss_coeff", 0.05, 1),
        "replay_batch_size": hp.choice("replay_batch_size", [8192, 16384, 16384*2]),
  }
current_best_params = [
	{
		"fast_loss_coeff": 0.05,
    "replay_batch_size": 0
	},
  {
		"fast_loss_coeff": 0.05,
    "replay_batch_size": 1
	},
  {
		"fast_loss_coeff": 0.05,
    "replay_batch_size": 2
	},
  {
		"fast_loss_coeff": 0.25,
    "replay_batch_size": 0
	},
  {
		"fast_loss_coeff": 0.25,
    "replay_batch_size": 1
	},
  {
		"fast_loss_coeff": 0.25,
    "replay_batch_size": 2
	},
  {
		"fast_loss_coeff": 1,
    "replay_batch_size": 0
	},
  {
		"fast_loss_coeff": 1,
    "replay_batch_size": 1
	},
  {
		"fast_loss_coeff": 1,
    "replay_batch_size": 2
	},
]

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)