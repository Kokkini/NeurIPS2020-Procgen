"""
Registry of custom implemented algorithms names

Please refer to the following examples to add your custom algorithms : 

- AlphaZero : https://github.com/ray-project/ray/tree/master/rllib/contrib/alpha_zero
- bandits : https://github.com/ray-project/ray/tree/master/rllib/contrib/bandits
- maddpg : https://github.com/ray-project/ray/tree/master/rllib/contrib/maddpg
- random_agent: https://github.com/ray-project/ray/tree/master/rllib/contrib/random_agent

An example integration of the random agent is shown here : 
- https://github.com/AIcrowd/neurips2020-procgen-starter-kit/tree/master/algorithms/custom_random_agent
"""


def _import_custom_random_agent():
    from .custom_random_agent.custom_random_agent import CustomRandomAgent
    return CustomRandomAgent

def _import_random_policy():
    from .random_policy.trainer import RandomPolicyTrainer
    return RandomPolicyTrainer

def _import_my_ppo():
    from .my_ppo.ppo import MyPPOTrainer
    return MyPPOTrainer

def _import_beta_vae_ppo():
    from .beta_vae.ppo import BetaVaePPOTrainer
    return BetaVaePPOTrainer


def _import_beta_vae_ppo_test():
    from .beta_vae_test.ppo import BetaVaePPOTrainerTest
    return BetaVaePPOTrainerTest

def _import_fast_head():
    from .fast_head.ppo import FastHeadTrainer
    return FastHeadTrainer


CUSTOM_ALGORITHMS = {
    "custom/CustomRandomAgent": _import_custom_random_agent,
    "RandomPolicy": _import_random_policy,
    "MyPPO": _import_my_ppo,
    "BetaVaePPO": _import_beta_vae_ppo,
    "BetaVaePPOTest": _import_beta_vae_ppo_test,
    "FastHead": _import_fast_head,
}
