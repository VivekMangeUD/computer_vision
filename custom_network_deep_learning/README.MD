# Custom Convolutional Neural Network Project

**Author:** Vivek Mange

This project involves building a custom convolutional neural network (CNN) for image classification using the CIFAR100 dataset. The goal is to design, train, and test a network from scratch.

### Network Architecture Requirements:

1. **Convolutional Layers:** The network must include at least 3 convolutional layers to extract features from the input images.
2. **Fully Connected Layer:** Incorporate at least 1 fully connected layer to perform classification based on the features.
3. **Max-Pooling Layer:** Include at least 1 max-pooling layer to reduce the spatial dimensions of the feature maps.
4. **Activation Function:** Use ReLU (Rectified Linear Unit) as the activation function for all layers.

### Training:

- **Dataset:** CIFAR100
- **Training from Scratch:** Since this is a new architecture, no pre-trained model is available. The network must be trained from scratch.
- **Epochs:** Train the network for at least 30 epochs.
- **Save Best Model:** Save the model that achieves the highest accuracy during training.

### Testing:

1. **Load Best Model:** After training, load the best model for testing.
2. **Predict Labels:** Use the model to predict the labels for the test set.
3. **Print Accuracy:** Calculate and print the testing accuracy of the model.

### Output:

- **Best Model:** Saved as a file containing the weights of the model that performed best during training.
- **Testing Accuracy:** A metric representing the model's performance on the test set.

