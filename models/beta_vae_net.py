from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

tf = try_import_tf()

from functools import partial
import numpy as np

conv = partial(tf.keras.layers.Conv2D, activation=None)
dconv = partial(tf.keras.layers.Conv2DTranspose, activation=None)
fc = partial(tf.keras.layers.Dense, activation=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
batch_norm = partial(tf.keras.layers.BatchNormalization, scale=True, updates_collections=None)


def conv_layer(depth, name):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name
    )

def conv_transpose_layer(depth, name):
    return tf.keras.layers.Conv2DTranspose(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name
    )

def conv_64():

    def Enc(img, z_dim, dim=64, is_training=True):
        bn = partial(batch_norm, is_training=is_training)
        conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation=lrelu)

        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = conv_bn_lrelu(img, dim, 5, 2)
            y = conv_bn_lrelu(y, dim * 2, 5, 2)
            y = conv_bn_lrelu(y, dim * 4, 5, 2)
            y = conv_bn_lrelu(y, dim * 8, 5, 2)
            z_mu = fc(y, z_dim)
            z_log_sigma_sq = fc(y, z_dim)
            return z_mu, z_log_sigma_sq

    def Dec(z, dim=64, channels=3, is_training=True):
        bn = partial(batch_norm, is_training=is_training)
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation=relu)

        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = relu(fc(z, 4 * 4 * dim * 8))
            y = tf.reshape(y, [-1, 4, 4, dim * 8])
            y = dconv_bn_relu(y, dim * 4, 5, 2)
            y = dconv_bn_relu(y, dim * 2, 5, 2)
            y = dconv_bn_relu(y, dim * 1, 5, 2)
            img = tf.tanh(dconv(y, channels, 5, 2))
            return img

    return Enc, Dec

def residual_block(x, depth, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == depth
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv1")(x)
    return x + inputs

def reverse_residual_block(x, depth, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == depth
    x = conv_transpose_layer(depth, name=prefix + "_reverse_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_transpose_layer(depth, name=prefix + "_reverse_conv1")(x)
    x = tf.keras.layers.ReLU()(x)
    return x + inputs

def conv_sequence(x, depth, prefix):
    x = conv_layer(depth, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
    return x

def reverse_conv_sequence(x, depth, prefix):
    x = reverse_residual_block(x, depth, prefix=prefix + "_reverse_block0")
    x = reverse_residual_block(x, depth, prefix=prefix + "_reverse_block1")
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = conv_transpose_layer(depth, prefix + "_reverse_conv")(x)
    return x


class BetaVaeNet(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        depths = model_config.get("custom_options", {}).get("depths", [16, 32, 32])
        self.original_shape = obs_space.shape
        print("obs space:", obs_space.shape)
        channels = obs_space.shape[-1]
        self.pad_shape = [72, 72, channels]
        shrink_factor = 2**len(depths)
        before_z_dim = np.array([self.pad_shape[0]//shrink_factor, self.pad_shape[1]//shrink_factor, depths[-1]])
        reverse_depths = depths[::-1][1:] + [channels]


        inputs = tf.keras.layers.Input(shape=self.pad_shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs
        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, prefix=f"seq{i}")



        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        z_dim = 256
        z_mu = tf.keras.layers.Dense(units=z_dim)(x)
        z_log_sigma_sq = tf.keras.layers.Dense(units=z_dim)(x)

        epsilon = tf.random_normal(tf.shape(z_mu))
        z = z_mu + tf.exp(0.5 * z_log_sigma_sq) * epsilon

        x_dec = z
        x_dec = tf.keras.layers.Dense(units=np.prod(before_z_dim))(x_dec)
        x_dec = tf.keras.layers.RELU()(x_dec)
        x_dec = tf.reshape(x_dec, [-1]+list(before_z_dim))
        for i, depth in enumerate(reverse_depths):
            x_dec = reverse_conv_sequence(x_dec, depth, prefix=f"reverse_seq{i}")
        x = z
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)
        self.base_model = tf.keras.Model(inputs, [logits, value, scaled_inputs, x_dec, z_mu, z_log_sigma_sq])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        obs = tf.image.pad_to_bounding_box(obs,
                                    tf.random.uniform(shape=[], minval=0,
                                                      maxval=self.pad_shape[0] - 64,
                                                      dtype=tf.int64),
                                    tf.random.uniform(shape=[], minval=0,
                                                      maxval=self.pad_shape[1] - 64,
                                                      dtype=tf.int64),
                                    self.pad_shape[0], self.pad_shape[1])
        logits, self._value, self.img, self.img_dec, self.z_mu, self.z_log_sigma_sq = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


# Register model in ModelCatalog
ModelCatalog.register_custom_model("beta_vae_net", BetaVaeNet)
