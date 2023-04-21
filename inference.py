import torch
import numpy as np
from PIL import Image
from cnn import CNNModel

# pass in the path of CT scan here to make prediction
img_path = 'test_set/covid/covid0.png'
img = Image.open(img_path).resize((150, 150))

# convert PIL image to a numpy array and normalize
img_array = np.array(img).astype('float32')
img_array = np.transpose(img_array, (2, 0, 1))  # Change shape to (channels, height, width)
img_array = np.expand_dims(img_array, axis=0)

# convert numpy array to a PyTorch tensor
img_tensor = torch.from_numpy(img_array)

# load the saved weights here to recreate the model
state_dict = torch.load("savedmodel/trained_model.pth")
reconstructed_model = CNNModel()
reconstructed_model.load_state_dict(state_dict)
reconstructed_model.eval()

# make prediction on image tensor
with torch.no_grad():
    output = reconstructed_model(img_tensor)
    prob = torch.sigmoid(output).item()

# Compare the predicted probability to the threshold
threshold = 0.5
if prob > threshold:
    predicted = 'Positive'
else:
    predicted = 'Negative'

print('Predicted result:', predicted)