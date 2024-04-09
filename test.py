import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

import model



DATA_DIR = "data2"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = "models"
BEST_PATH = os.path.join(MODEL_DIR, "best2.pt")

BATCH_SIZE = 10

# Create a mapping between numeric labels and class names
class_mapping = {1: "Cats", 2: "Dogs"}  

# Define the test data transformations
test_transform = transforms.Compose([
    transforms.CenterCrop(model.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
vit = model.VisionTransformer(num_classes=len(test_dataset.classes), device=device)
vit.load_state_dict(torch.load(BEST_PATH))
vit.eval()

# Move the model to the device (GPU or CPU)
vit.to(device)

# Perform inference on the test data
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vit(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predicted_classes = [idx_to_class[label.item()] for label in predicted]
        actual_classes = [idx_to_class[label.item()] for label in labels]

        # Print the predicted and actual class names
        print("Predicted:", predicted_classes)
        print("Actual:   ", actual_classes)
    
# Calculate and print the accuracy
accuracy = float(correct) / float(total)
print(f"Test Accuracy: {accuracy:.4f}")
