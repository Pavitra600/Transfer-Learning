# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop a deep learning model for image classification using transfer learning. Utilize the pre-trained VGG19 model as the feature extractor, fine-tune it, and adapt it to classify images into specific categories.

## DESIGN STEPS
### **Step 1: Import Libraries and Load Dataset**
- Import the necessary libraries.
- Load the dataset.
- Split the dataset into training and testing sets.

### **Step 2: Initialize Model, Loss Function, and Optimizer**
- Define the model architecture.
- Use `CrossEntropyLoss` for multi-class classification.
- Choose the `Adam` optimizer for efficient training.

### **Step 3: Train the Model**
- Train the model using the training dataset.
- Optimize the model parameters to minimize the loss.

### **Step 4: Evaluate the Model**
- Test the model using the testing dataset.
- Measure performance using appropriate evaluation metrics.

### **Step 5: Make Predictions on New Data**
- Use the trained model to predict outcomes for new inputs./>


## PROGRAM
```
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False


# Modify the final fully connected layer to match one binary classes
num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features,1)


# Include the Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: Dharshini S N")
    print("Register Number:212224230062")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="690" height="640" alt="image" src="https://github.com/user-attachments/assets/2e578651-2963-4501-8c27-de2cd0aed81e" />

</br>
</br>
</br>

### Confusion Matrix
<img width="612" height="568" alt="image" src="https://github.com/user-attachments/assets/d39e6c67-4291-4330-8d4d-6f0247fea1bb" />


</br>
</br>
</br>

### Classification Report
<img width="463" height="198" alt="image" src="https://github.com/user-attachments/assets/8da9957f-26f9-44dd-8667-e4bce5190df9" />

</br>
</br>
</br>

### New Sample Prediction
<img width="651" height="645" alt="image" src="https://github.com/user-attachments/assets/1442e9eb-f4a2-4890-ac18-b3c13e2cb66a" />


</br>
</br>
</br>

## RESULT
Thus, the transfer Learning for classification using VGG-19 architecture has succesfully implemented.
