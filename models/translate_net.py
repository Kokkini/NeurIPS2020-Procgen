from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

tf = try_import_tf()


def conv_layer(depth, regularizer, name):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name, kernel_regularizer=regularizer
    )


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
    if regu > 0:
        regularizer = tf.keras.regularizers.L2(l2=regu)
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


class TranslateNet(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        depths = model_config.get("custom_options", {}).get("depths", [16, 32, 32])
        regu = model_config.get("custom_options", {}).get("regu", 0)
        self.original_shape = obs_space.shape
        print("obs space:", obs_space.shape)
        channels = obs_space.shape[-1]
        self.pad_shape = [72, 72, channels]

        inputs = tf.keras.layers.Input(shape=self.pad_shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs
        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, regu, prefix=f"seq{i}")

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)
        self.base_model = tf.keras.Model(inputs, [logits, value])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        # random padding
        # obs = tf.map_fn(lambda image: tf.image.pad_to_bounding_box(image,
        #                                                            tf.random.uniform(shape=[], minval=0,
        #                                                                              maxval=self.pad_shape[0] - 64,
        #                                                                              dtype=tf.int64),
        #                                                            tf.random.uniform(shape=[], minval=0,
        #                                                                              maxval=self.pad_shape[1] - 64,
        #                                                                              dtype=tf.int64),
        #                                                            self.pad_shape[0], self.pad_shape[1]), obs)
        obs = tf.image.pad_to_bounding_box(obs,
                                    tf.random.uniform(shape=[], minval=0,
                                                      maxval=self.pad_shape[0] - 64,
                                                      dtype=tf.int64),
                                    tf.random.uniform(shape=[], minval=0,
                                                      maxval=self.pad_shape[1] - 64,
                                                      dtype=tf.int64),
                                    self.pad_shape[0], self.pad_shape[1])
        logits, self._value = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


# Register model in ModelCatalog
ModelCatalog.register_custom_model("translate_net", TranslateNet)
