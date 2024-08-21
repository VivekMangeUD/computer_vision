# Semantic Segmentation Project

**Author:** Vivek Mange

This project involves using a Fully Convolutional Network (FCN) for semantic segmentation, focusing on identifying and segmenting objects within images from 21 different classes.

### Tasks:

1. **Download Images:** Obtain 5 or more images, each containing at least 3 objects from the following classes: Aeroplane, Bicycle, Bird, Boat, Bottle, Bus, Car, Cat, Chair, Cow, Dining Table, Dog, Horse, Motorbike, Person, Potted Plant, Sheep, Sofa, Train, TV/Monitor.

2. **Load Pre-trained FCN:** Use PyTorch to load a pre-trained FCN network, either `fcn_resnet50` or `fcn_resnet101`.

3. **Image Processing:**
   - Feed each image into the network.
   - The network will output 21 feature maps, each corresponding to a class.
   - Save all feature maps as tiles in a single image.

4. **Segmentation Creation:**
   - Create a final segmentation image where each class is represented by a unique color.
   - Experiment with different input sizes to achieve the best segmentation results.

### Output:
- **Feature Maps:** Saved as a tiled image.
- **Segmentation Image:** Color-coded image representing the segmented classes.

