import tensorflow as tf


class DataAugmentor:

    def __init__(self, config, seed):
        """
        Arguments:
            config (dict): config for all augmentations.
            seed (int): default seed for probabilities and augmentations.
                seed for augmentation could be overwritten by config.
        """
        self._config = config
        self._seed = seed
        tf.random.set_seed(self._seed)  # set seed for probabilities.

    def call(self, x):
        for augmentation in self._config:
            config = self._config[augmentation]
            rand = tf.random.uniform(shape=(1,)).numpy()[0]
            if augmentation == 'random_zoom':
                if rand < config.get('probability', 1.0):
                    x = tf.keras.layers.RandomZoom(
                        height_factor=(self._config['random_zoom']['lower'], self._config['random_zoom']['upper']),
                        seed=config.get('seed', self._seed),
                    )(x)
            elif augmentation == 'random_translation':
                if rand < config.get('probability', 1.0):
                    x = tf.keras.layers.RandomTranslation(
                        height_factor=config['x_max'], width_factor=config['y_max'],
                        seed=config.get('seed', self._seed),
                    )(x)
            elif augmentation == 'random_brightness':
                if rand < config.get('probability', 1.0):
                    x = tf.image.stateless_random_brightness(
                        image=x, max_delta=config['max_delta'],
                    )
            elif augmentation == 'random_saturation':
                if rand < config.get('probability', 1.0):
                    x = tf.image.stateless_random_saturation(
                        image=x, lower=config['lower'], upper=config['upper'],
                    )
            else:
                raise ValueError(f"[ERROR] Augmentation name \"{augmentation}\" not implemented.")
        return x
