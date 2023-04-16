import tensorflow as tf

class ImageClassificationModel(tf.keras.Model):
    def __init__(self):
        super(ImageClassificationModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.dense1 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        outputs = self.dense2(x)
        return outputs
