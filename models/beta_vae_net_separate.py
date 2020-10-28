from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog
import numpy as np

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


def get_enc_dec(z_dim, depths, before_z_dim, channels):
    def Enc(img):
        # bn = partial(batch_norm, is_training=is_training)
        y = img
        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            for d in depths:
                y = tf.keras.layers.Conv2D(filters=d, activation="relu", kernel_size=4, strides=2, padding="same")(y)
            y = tf.keras.layers.Flatten()(y)
            y = tf.keras.layers.Dense(256, activation="relu")(y) 
            z_mu = tf.keras.layers.Dense(units=z_dim)(y)
            z_log_sigma_sq = tf.keras.layers.Dense(units=z_dim)(y)
            return z_mu, z_log_sigma_sq

    def Dec(z):
        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = z
            y = tf.keras.layers.Dense(256, activation="relu")(y)
            y = tf.keras.layers.Dense(units=np.prod(before_z_dim), activation="relu")(y)
            y = tf.reshape(y, [-1]+list(before_z_dim))
            reverse_depths = depths[::-1]
            for d in reverse_depths:
                y = tf.keras.layers.Conv2DTranspose(filters=d, activation="relu", kernel_size=4, strides=2, padding="same")(y)
            img = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=4, strides=1, padding="same")(y)
            return img
    return Enc, Dec

class BetaVaeNetSeparate(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        depths = model_config.get("custom_options", {}).get("depths", [16, 32, 32])
        vae_depths = model_config.get("custom_options", {}).get("vae_depths", [16, 32, 32])
        z_dim = model_config.get("custom_options", {}).get("z_dim", 100)
        regu = model_config.get("custom_options", {}).get("regu", 0)
        vae_grad = model_config.get("custom_options", {}).get("vae_grad", True)
        vae_norm = model_config.get("custom_options", {}).get("vae_norm", False)
        use_vae_features = model_config.get("custom_options", {}).get("use_vae_features", True)
        use_impala_features = model_config.get("custom_options", {}).get("use_impala_features", True)
        dense_regu = model_config.get("custom_options", {}).get("dense_regu", 0)
        # dense_depths = model_config.get("custom_options", {}).get("dense_depths", [256])
        self.original_shape = obs_space.shape
        print("obs space:", obs_space.shape)
        channels = obs_space.shape[-1]
        self.pad_shape = [72, 72, channels]
        shrink_factor = 2**len(vae_depths)
        before_z_dim = np.array([self.pad_shape[0]//shrink_factor, self.pad_shape[1]//shrink_factor, vae_depths[-1]])

        enc, dec = get_enc_dec(z_dim, vae_depths, before_z_dim, channels)

        inputs = tf.keras.layers.Input(shape=self.pad_shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs
        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, regu, prefix=f"seq{i}")

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)
        
        z_mu, z_log_sigma_sq = enc(scaled_inputs)
        epsilon = tf.random_normal(tf.shape(z_mu))
        z = z_mu + tf.exp(0.5 * z_log_sigma_sq) * epsilon

        x_dec = dec(z)

        z_mu_1 = z_mu
        if not vae_grad:
            z_mu_1 = tf.stop_gradient(z_mu)
        
        if vae_norm:
            x = tf.keras.layers.LayerNormalization()(x)
            z_mu_1 = tf.keras.layers.LayerNormalization()(z_mu_1)

        impala_base = x

        if use_impala_features and use_vae_features:
            x = tf.concat([x, z_mu_1], axis=1)
        elif use_impala_features:
            x = x
        elif use_vae_features:
            x = z_mu_1
        else:
            print("at least one of use_vae_features or use_impala_features must be True")
            exit(1)
        
        regularizer = None
        if dense_regu > 0:
            regularizer = tf.keras.regularizers.l1(l=dense_regu)
        x = tf.keras.layers.Dense(256, kernel_regularizer=regularizer)(x)

        # for ix in range(1, len(dense_depths)):
        #     x = tf.keras.layers.Dense(units=dense_depths[ix], activation="relu")(x)


        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)
        self.base_model = tf.keras.Model(inputs, [logits, value, scaled_inputs, x_dec, z_mu, z_log_sigma_sq])
        self.register_variables(self.base_model.variables)
        if use_impala_features == False:
            impala_base_model = tf.keras.Model(inputs, [impala_base])
            self.register_variables(impala_base_model.variables)

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
ModelCatalog.register_custom_model("beta_vae_net_separate", BetaVaeNetSeparate)
