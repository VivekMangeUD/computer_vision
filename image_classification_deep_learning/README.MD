# Image Classification Project

**Author:** Vivek Mange

This project involves training a VGG16 network on the CIFAR100 dataset for image classification. The code is partially provided, and the following steps guide you through completing and executing the tasks.

### Tasks:

1. **Import Libraries:** Import essential libraries like PyTorch, Torchvision, and others required for the project.

2. **Image Preprocessing:** Resize images to (224, 224), convert them to Tensors, and normalize RGB values.

3. **Dataset Loading:** Use `torchvision.datasets.CIFAR100` to load the CIFAR100 dataset with transformations applied.

4. **Data Loader:** Create a data loader for batch processing of the dataset.

5. **Model Setup:** Load a pre-trained VGG16 model, modify the last layer for CIFAR100, and freeze other layers.

6. **Training Configuration:** Set the number of epochs, loss function, optimizer, and learning rate scheduler.

7. **Training Loop:** Implement a loop to train the model, updating weights, and saving the best model.

8. **Model Testing:** Load the best model, set it to evaluation mode, and calculate the accuracy on the test set.
