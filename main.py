import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from cnn import CNNModel


# Directories
train_dir = './train_set/'
test_dir = './test_set/'

# Image transformations for train and test sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Instantiate the model
model = CNNModel()
print(model)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 500
training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        preds = torch.round(outputs)
        running_corrects += torch.sum(preds == labels.data.float())

    # Calculate average loss and accuracy
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects / len(train_dataset)

    training_losses.append(epoch_loss)
    training_accuracies.append(epoch_acc.item())

    # Validation loop
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0

    with torch.no_grad():
        for val_inputs, val_labels in test_loader:
            val_outputs = model(val_inputs)
            val_outputs = val_outputs.view(-1)
            val_loss = criterion(val_outputs, val_labels.float())

            val_running_loss += val_loss.item()

            # Compute validation accuracy
            val_preds = torch.round(val_outputs)
            val_running_corrects += torch.sum(val_preds == val_labels.data.float())

        # Calculate average validation loss and accuracy
        val_epoch_loss = val_running_loss / len(test_loader)
        val_epoch_acc = val_running_corrects / len(test_dataset)

        validation_losses.append(val_epoch_loss)
        validation_accuracies.append(val_epoch_acc.item())

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val_Loss: {val_epoch_loss:.4f}, Val_Acc: {val_epoch_acc:.4f}')

torch.save(model.state_dict(), './savedmodel/trained_model.pth')

# Plot training and validation accuracy
plt.figure()
plt.plot(range(1, num_epochs + 1), training_accuracies, 'ro', label='Training acc')
plt.plot(range(1, num_epochs + 1), validation_accuracies, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')

# Plot training and validation loss
plt.figure()
plt.plot(range(1, num_epochs + 1), training_losses, 'ro', label='Training loss')
plt.plot(range(1, num_epochs + 1), validation_losses, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')

