from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog
import numpy as np

tf = try_import_tf()


def conv_layer(depth, regularizer, name):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name, kernel_regularizer=regularizer
    )


def conv_layer_stride(depth, stride):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=stride, padding="same")

def residual_block(x, depth, regularizer, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == depth
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, regularizer, name=prefix + "_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, regularizer, name=prefix + "_conv1")(x)
    return x + inputs


def conv_sequence(x, depth, regu, prefix):
    regularizer = None
    if regu:
        regularizer = tf.keras.regularizers.L2(l2=0.01)
    x = conv_layer(depth, regularizer, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, regularizer, prefix=prefix + "_block0")
    x = residual_block(x, depth, regularizer, prefix=prefix + "_block1")
    return x


def random_crop(obs, is_training, crop_shape, original_shape):
    if is_training:
        return tf.map_fn(lambda image: tf.image.random_crop(image, crop_shape), obs)
    else:
        y1 = (original_shape[0] - crop_shape[0]) // 2
        x1 = (original_shape[1] - crop_shape[1]) // 2
        h = crop_shape[0]
        w = crop_shape[1]
        return tf.image.crop_to_bounding_box(obs, y1, x1, h, w)

def fast_head_pass(x, num_outputs, trainable=True):
    x = tf.keras.layers.Dense(units=256, trainable=trainable)(x)
    x = tf.keras.layers.Dense(units=num_outputs, trainable=trainable)(x)
    return x


class FastHeadNet(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        depths = model_config.get("custom_options", {}).get("depths", [16, 32, 32])
        regu = model_config.get("custom_options", {}).get("regu", False)
        self.original_shape = obs_space.shape
        print("obs space:", obs_space.shape)
        channels = obs_space.shape[-1]
        self.pad_shape = [72, 72, channels]

        inputs = tf.keras.layers.Input(shape=self.pad_shape, name="observations")

        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs
        # for i, depth in enumerate(depths):
        #     x = conv_sequence(x, depth, regu, prefix=f"seq{i}")
        for i, depth in enumerate(depths):
            x = conv_layer_stride(depth, 2)(x)

        x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units=256, name="hidden")(x)

        embedding = x
        
        fast_head_1 = tf.keras.layers.Dense(units=256,activation='relu')
        fast_head_2 = tf.keras.layers.Dense(units=256,activation='relu')
        fast_head_3 = tf.keras.layers.Dense(units=num_outputs)

        logits = fast_head_3(fast_head_2(fast_head_1(x)))

        # value = tf.keras.layers.Dense(units=1, name="vf")(x)
        self.base_model = tf.keras.Model(inputs, [logits, embedding])

        fast_head_inputs = tf.keras.layers.Input(shape=[256], name="embedding")
        fast_head_out = fast_head_3(fast_head_2(fast_head_1(fast_head_inputs)))

        self.fast_head_model = tf.keras.Model(fast_head_inputs, fast_head_out)

        self.register_variables(self.base_model.variables + self.fast_head_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        subtract_mask = np.zeros([1] + list(self.original_shape))
        subtract_mask[...,3:] = 1
        subtract_part = subtract_mask * obs
        subtract_part = subtract_part * 2 - 255 + 1   #plus 1 to make the background 0 instead of -1
        obs = obs * (1-subtract_mask) + subtract_part * subtract_mask

        obs = tf.image.pad_to_bounding_box(obs,
                                    tf.random.uniform(shape=[], minval=0,
                                                      maxval=self.pad_shape[0] - 64,
                                                      dtype=tf.int64),
                                    tf.random.uniform(shape=[], minval=0,
                                                      maxval=self.pad_shape[1] - 64,
                                                      dtype=tf.int64),
                                    self.pad_shape[0], self.pad_shape[1])
        self.logits, self.embedding = self.base_model(obs)
        return self.logits, state

    def value_function(self):
        return tf.reduce_max(self.logits, axis=1)


# Register model in ModelCatalog
ModelCatalog.register_custom_model("fast_head_net", FastHeadNet)
