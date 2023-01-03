import tensorflow as tf


class Rewrite(tf.keras.Model):

    def __init__(self, char_size=80):
        input = tf.keras.layers.Input(shape=(char_size, char_size, 3))
        super(Rewrite, self).__init__(
            name='rewrite', inputs=input, outputs=tf.keras.layers.ReLU()(input))

        x = tf.keras.layers.Conv2D(
            filters=8, kernel_size=(64, 64), padding='same')(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=8, kernel_size=(64, 64), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(32, 32), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(32, 32), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(32, 32), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(32, 32), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(16, 16), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(16, 16), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(16, 16), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(16, 16), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(7, 7), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(7, 7), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(7, 7), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(7, 7), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=3, kernel_size=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        output = tf.keras.layers.Activation('sigmoid')(x)

        self.model = tf.keras.Model(inputs=input, outputs=output)
        super(Rewrite, self).__init__(
            name='rewrite', inputs=input, outputs=output)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)


if __name__ == '__main__':
    model = Rewrite()
    model.summary()
