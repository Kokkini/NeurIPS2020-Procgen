import logging

import ray
from .postprocessing import compute_advantages, Postprocessing
from .sample_batch import SampleBatch
from .tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from .tf_policy_template import build_tf_policy
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils import try_import_tf
import random
import numpy as np
from collections import deque

tf = try_import_tf()

logger = logging.getLogger(__name__)

class EmbeddingBuffer:
    def __init__(self, N):
        self.embedding = deque()
        self.action = deque()
        self.q = deque()
        self.N = N


buffer = EmbeddingBuffer(1000000)

# class PPOLoss:
#     def __init__(self,
#                  dist_class,
#                  model,
#                  value_targets,
#                  advantages,
#                  actions,
#                  prev_logits,
#                  prev_actions_logp,
#                  vf_preds,
#                  embeddings,
#                  curr_action_dist,
#                  value_fn,
#                  cur_kl_coeff,
#                  valid_mask,
#                  entropy_coeff=0,
#                  fast_loss_coeff=1,
#                  clip_param=0.1,
#                  vf_clip_param=0.1,
#                  vf_loss_coeff=1.0,
#                  use_gae=True,
#                  down_value_weight=None,
#                  mini_batch_size=256,
#                  fast_head_batch_multiplier=10):
#         """Constructs the loss for Proximal Policy Objective.

#         Arguments:
#             dist_class: action distribution class for logits.
#             value_targets (Placeholder): Placeholder for target values; used
#                 for GAE.
#             actions (Placeholder): Placeholder for actions taken
#                 from previous model evaluation.
#             advantages (Placeholder): Placeholder for calculated advantages
#                 from previous model evaluation.
#             prev_logits (Placeholder): Placeholder for logits output from
#                 previous model evaluation.
#             prev_actions_logp (Placeholder): Placeholder for action prob output
#                 from the previous (before update) Model evaluation.
#             vf_preds (Placeholder): Placeholder for value function output
#                 from the previous (before update) Model evaluation.
#             curr_action_dist (ActionDistribution): ActionDistribution
#                 of the current model.
#             value_fn (Tensor): Current value function output Tensor.
#             cur_kl_coeff (Variable): Variable holding the current PPO KL
#                 coefficient.
#             valid_mask (Optional[tf.Tensor]): An optional bool mask of valid
#                 input elements (for max-len padded sequences (RNNs)).
#             entropy_coeff (float): Coefficient of the entropy regularizer.
#             clip_param (float): Clip parameter
#             vf_clip_param (float): Clip parameter for the value function
#             vf_loss_coeff (float): Coefficient of the value function loss
#             use_gae (bool): If true, use the Generalized Advantage Estimator.
#         """
#         if valid_mask is not None:

#             def reduce_mean_valid(t):
#                 return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

#         else:

#             def reduce_mean_valid(t):
#                 return tf.reduce_mean(t)

#         # Make loss functions.

#         slow_loss = reduce_mean_valid(tf.square(value_fn - value_targets))
#         fast_loss = 0
#         fast_head_batch_size = fast_head_batch_multiplier * mini_batch_size
#         if len(buffer.action) > max(100000, fast_head_batch_size):
#             sampled_ix = np.random.choice(len(buffer.action), fast_head_batch_size, replace=False)
#             sampled_embedding = [buffer.embdding[j] for j in sampled_ix] + embeddings
#             sampled_action = [buffer.action[j] for j in sampled_ix] + actions
#             sampled_q = [buffer.q[j] for j in sampled_ix] + value_targets

#             fast_head_out = self.fast_head_model(tf.convert_to_tensor(sampled_embedding))
#             fast_head_value = tf.one_hot(tf.convert_to_tensor(sampled_action), tf.shape(fast_head_out)[-1]) * fast_head_out
#             fast_loss =  reduce_mean_valid(tf.square(fast_head_out - tf.convert_to_tensor(sampled_q)))

#         buffer.q.extend(value_targets)
#         buffer.action.extend(actions)
#         buffer.embedding.extend(embeddings)
        
        
#         for j in range(len(buffer.embedding) - buffer.N):
#             buffer.embedding.popleft()
#             buffer.q.popleft()
#             buffer.action.popleft()
#         assert len(buffer.q) == len(buffer.action) == len(buffer.embedding)

#         self.loss = slow_loss + fast_loss_coeff * fast_loss


# def ppo_surrogate_loss(policy, model, dist_class, train_batch):
#     logits, state = model.from_batch(train_batch)
#     action_dist = dist_class(logits, model)

#     mask = None
#     if state:
#         max_seq_len = tf.reduce_max(train_batch["seq_lens"])
#         mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
#         mask = tf.reshape(mask, [-1])

#     policy.loss_obj = PPOLoss(
#         dist_class,
#         model,
#         train_batch[Postprocessing.VALUE_TARGETS],
#         train_batch[Postprocessing.ADVANTAGES],
#         train_batch[SampleBatch.ACTIONS],
#         train_batch[SampleBatch.ACTION_DIST_INPUTS],
#         train_batch[SampleBatch.ACTION_LOGP],
#         train_batch[SampleBatch.VF_PREDS],
#         train_batch[SampleBatch.EMBEDDING],
#         action_dist,
#         model.value_function(),
#         policy.kl_coeff,
#         mask,
#         entropy_coeff=policy.entropy_coeff,
#         fast_loss_coeff=policy.config["fast_loss_coeff"],
#         clip_param=policy.config["clip_param"],
#         vf_clip_param=policy.config["vf_clip_param"],
#         vf_loss_coeff=policy.config["vf_loss_coeff"],
#         use_gae=policy.config["use_gae"],
#         down_value_weight=policy.config["down_value_weight"],
#         mini_batch_size=policy.config["sgd_minibatch_size"],
#         fast_head_batch_multiplier=policy.config["fast_head_batch_multiplier"],
#     )

#     return policy.loss_obj.loss

class PPOLoss:
    def __init__(self,
                 dist_class,
                 model,
                 train_batch,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 fast_loss_coeff=1,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True,
                 down_value_weight=None):

        self.loss = tf.cond(tf.reduce_all(train_batch[Postprocessing.FROM_BUFFER]),
                            lambda : fast_loss_coeff * self.get_fast_loss(model, train_batch),
                            lambda : get_slow_loss(model, train_batch))

    def get_slow_loss(self, model, train_batch):
        logits, state = model.from_batch(train_batch)
        value_fn = model.value_function()
        value_targets = train_batch[Postprocessing.VALUE_TARGETS]
        return tf.reduce_mean(tf.square(value_fn - value_targets))

    def get_fast_loss(self, model, train_batch):
        # fast_head_out = self.model.fast_head_model(embeddings)
        embeddings = train_batch[SampleBatch.EMBEDDING]
        fast_head_out = model.fast_head_model(embeddings)
        actions = train_batch[SampleBatch.ACTIONS]
        value_targets = train_batch[Postprocessing.VALUE_TARGETS]
        fast_head_value = tf.reduce_sum(tf.one_hot(actions, tf.shape(fast_head_out)[-1]) * fast_head_out, axis=1)
        fast_loss = tf.reduce_mean(tf.square(fast_head_value - value_targets))
        return fast_loss
        


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    

    mask = None

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch,
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        fast_loss_coeff=policy.config["fast_loss_coeff"],
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        down_value_weight=policy.config["down_value_weight"])

    return policy.loss_obj.loss

def kl_and_loss_stats(policy, train_batch):
    return {
        # "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        # "policy_loss": policy.loss_obj.mean_policy_loss,
        # "vf_loss": policy.loss_obj.mean_vf_loss,
        # "vf_explained_var": explained_variance(
        #     train_batch[Postprocessing.VALUE_TARGETS],
        #     policy.model.value_function()),
        # "kl": policy.loss_obj.mean_kl,
        # "entropy": policy.loss_obj.mean_entropy,
        # "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }


def vf_preds_fetches(policy):
    """Adds value function outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        SampleBatch.EMBEDDING: policy.model.embedding,
    }


def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch


def clip_gradients(policy, optimizer, loss):
    variables = policy.model.trainable_variables()
    if policy.config["grad_clip"] is not None:
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        grads = [g for (g, v) in grads_and_vars]
        policy.grads, _ = tf.clip_by_global_norm(grads,
                                                 policy.config["grad_clip"])
        clipped_grads = list(zip(policy.grads, variables))
        return clipped_grads
    else:
        return optimizer.compute_gradients(loss, variables)


class KLCoeffMixin:
    def __init__(self, config):
        # KL Coefficient
        self.kl_coeff_val = config["kl_coeff"]
        self.kl_target = config["kl_target"]
        self.kl_coeff = tf.get_variable(
            initializer=tf.constant_initializer(self.kl_coeff_val),
            name="kl_coeff",
            shape=(),
            trainable=False,
            dtype=tf.float32)

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff_val *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff_val *= 0.5
        self.kl_coeff.load(self.kl_coeff_val, session=self.get_session())
        return self.kl_coeff_val


class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        @make_tf_callable(self.get_session())
        def value(ob, prev_action, prev_reward, *state):
            model_out, _ = self.model({
                SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                    [prev_action]),
                SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                    [prev_reward]),
                "is_training": tf.convert_to_tensor(False),
            }, [tf.convert_to_tensor([s]) for s in state],
                                        tf.convert_to_tensor([1]))
            return self.model.value_function()[0]

        self._value = value


def setup_config(policy, obs_space, action_space, config):
    # auto set the model option for layer sharing
    config["model"]["vf_share_layers"] = config["vf_share_layers"]


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])

# def action_distribution_fn(policy, model, obs_batch, state_batches, seq_lens, prev_action_batch,
#                       prev_reward_batch, explore, is_training):
#     logits, state = model.from_batch(train_batch)
#     action_dist = dist_class(logits, model)
    
#     return dist_inputs, dist_class, self._state_out

PPOTFPolicy = build_tf_policy(
    name="PPOTFPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_fetches,
    postprocess_fn=postprocess_ppo_gae,
    # gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])
