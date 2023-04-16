import tensorflow as tf
import numpy as np

# pass in the path of CT scan here to make prediction
img = tf.keras.preprocessing.image.load_img('test_set/covid/covid0.png', target_size=(150, 150))

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# load the saved weights here to recreate the model
reconstructed_model = tf.keras.models.load_model("savedmodel/ourmodel")
prob = reconstructed_model.predict(img_array)[0][0]

# Compare the predicted probability to the threshold
threshold = 0.5
if prob > threshold:
    predicted = 'Positive'
else:
    predicted = 'Negative'

print('Predicted result:', predicted)

