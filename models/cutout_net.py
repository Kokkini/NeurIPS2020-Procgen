from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog
import tensorflow_addons as tfa

tf = try_import_tf()


def conv_layer(depth, name):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name
    )


def residual_block(x, depth, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == depth
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv1")(x)
    return x + inputs


def conv_sequence(x, depth, prefix):
    x = conv_layer(depth, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
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




class CutoutNet(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        depths = [16, 32, 32]
        crop_frac = 58 / 64
        self.original_shape = obs_space.shape
        print(self.original_shape)
        self.crop_shape = [int(crop_frac * obs_space.shape[0]), int(crop_frac * obs_space.shape[1]), 3]
        print(self.crop_shape)
        self.pad_shape = [74, 74, 3]

        self.cutout_min = 7
        self.cutout_max = 22
        self.cutout_chance = 0.8

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs
        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, prefix=f"seq{i}")

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)
        self.base_model = tf.keras.Model(inputs, [logits, value])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        print(input_dict, flush=True)
        # exit()
        obs = tf.cast(input_dict["obs"], tf.float32)
        if "is_training" in input_dict:
            is_training = input_dict["is_training"]
            if isinstance(is_training, tf.Tensor):
                obs = tf.cond(is_training, lambda: self.cutout_color(obs, True),
                              lambda: self.cutout_color(obs, False))
            elif is_training:
                obs = self.cutout_color(obs, True)
        logits, self._value = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])

    def random_cutout_color_one_image(self, img):
        p = tf.random.uniform([], minval=0, maxval=1)
        y = tf.random.uniform([], maxval=self.original_shape[0]-1, dtype=tf.int64)
        x = tf.random.uniform([], maxval=self.original_shape[1]-1, dtype=tf.int64)
        h = tf.random.uniform([], minval=self.cutout_min, maxval=self.cutout_max, dtype=tf.int64)
        w = tf.random.uniform([], minval=self.cutout_min, maxval=self.cutout_max, dtype=tf.int64)
        h = tf.minimum(h, self.original_shape[0])
        w = tf.minimum(w, self.original_shape[1])
        augmented = img.copy()
        augmented[y:y + h, x:x + w] = tf.random.uniform([h, w, 3], minval=0, maxval=255, dtype=tf.float32)
        res = tf.cond(p < self.cutout_chance, lambda: augmented, lambda: img)
        return res

    def cutout_color(self, obs, is_training):
        if is_training:
            return tf.map_fn(lambda image: self.random_cutout_color_one_image(image), obs)
        else:
            return obs



# Register model in ModelCatalog
ModelCatalog.register_custom_model("cutout_net", CutoutNet)
