import tensorflow as tf
import numpy as np

# Универсальный декоратор регистрации (Keras 2 / Keras 3)
try:
  register_keras_serializable = tf.keras.saving.register_keras_serializable
except AttributeError:
  register_keras_serializable = tf.keras.utils.register_keras_serializable

@register_keras_serializable()
class RandomReverse(tf.keras.layers.Layer):
  """С вероятностью 0.5 инвертирует временную ось (axis=1)."""
  def __init__(self, prob=0.5, **kwargs):
    super().__init__(**kwargs)
    self.prob = prob

  def call(self, inputs, training=None):
    if not training:
      return inputs
    batch_size = tf.shape(inputs)[0]
    mask = tf.random.uniform([batch_size]) < self.prob
    reversed_inputs = tf.reverse(inputs, axis=[1])
    return tf.where(mask[:, tf.newaxis, tf.newaxis],
                    reversed_inputs, inputs)

  def get_config(self):
    config = super().get_config()
    config.update({"prob": self.prob})
    return config

@register_keras_serializable()
class RandomRotation(tf.keras.layers.Layer):
  """Вращает всю последовательность на случайный угол."""
  def __init__(self, max_angle_deg=180.0, **kwargs):
    super().__init__(**kwargs)
    self.max_angle = np.deg2rad(max_angle_deg)

  def call(self, inputs, training=None):
    if not training:
      return inputs
    batch_size = tf.shape(inputs)[0]
    angle = tf.random.uniform([batch_size, 1, 1],
                              -self.max_angle, self.max_angle)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    x = inputs[..., 0:1]
    y = inputs[..., 1:2]
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    return tf.concat([x_rot, y_rot], axis=-1)

  def get_config(self):
    config = super().get_config()
    config.update({"max_angle_deg": np.rad2deg(self.max_angle)})
    return config

@register_keras_serializable()
class RandomAngleNoise(tf.keras.layers.Layer):
  """Добавляет независимый гауссов шум к углу каждой точки."""
  def __init__(self, stddev_rad=0.035, **kwargs):
    super().__init__(**kwargs)
    self.stddev = stddev_rad

  def call(self, inputs, training=None):
    if not training:
        return inputs
    batch_size = tf.shape(inputs)[0]
    timesteps = tf.shape(inputs)[1]
    dphi = tf.random.normal([batch_size, timesteps, 1],
                            mean=0.0, stddev=self.stddev)
    cos_d = tf.cos(dphi)
    sin_d = tf.sin(dphi)
    x = inputs[..., 0:1]
    y = inputs[..., 1:2]
    x_noisy = x * cos_d - y * sin_d
    y_noisy = x * sin_d + y * cos_d
    return tf.concat([x_noisy, y_noisy], axis=-1)

  def get_config(self):
    config = super().get_config()
    config.update({"stddev_rad": self.stddev})
    return config