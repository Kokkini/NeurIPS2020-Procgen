from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

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

def random_noise(obs, std):
    noise = tf.random.normal(tf.shape(obs), mean=0.0, stddev=std, dtype=tf.float32)
    return obs + noise


class NoiseNet(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.noise1 = tf.Variable(0, dtype=tf.float32)
        self.noise2 = tf.Variable(0, dtype=tf.float32)
        custom_options = model_config.get("custom_options")
        num_mini_batch = custom_options.get("num_mini_batch")
        print(model_config)
        if num_mini_batch is None:
            self.delta_noise = 0
        else:
            self.delta_noise = 20 / num_mini_batch
        print(f"num mini batch: {num_mini_batch}")

        depths = [16, 32, 32]
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
        self.register_variables([self.noise1, self.noise2])

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        print(input_dict, flush=True)
        # exit()
        obs = tf.cast(input_dict["obs"], tf.float32)
        if "is_training" in input_dict:
            is_training = input_dict["is_training"]
            if isinstance(is_training, tf.Tensor):
                obs = tf.cond(is_training, lambda: random_noise(obs, tf.minimum(20.0, self.noise1.assign_add(self.delta_noise))),
                              lambda: obs)
            elif is_training:
                obs = random_noise(obs, tf.minimum(20.0, self.noise2.assign_add(self.delta_noise)))

        logits, self._value = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


# Register model in ModelCatalog
ModelCatalog.register_custom_model("noise_net", NoiseNet)
