# Face Age Detection Using Convolutional Neural Network

The project implements a CNN model for the task of age detection using a dataset consisting of faces belonging to different age groups. There are 99 classes in the dataset which belong to different age groups and consist of around 10,000 images in total.

The dataset can be downloaded from [here](https://www.kaggle.com/datasets/frabbisw/facial-age).

### Description

The images are loaded into the code using the tensorflow function 'flow_from_dataset' and after loading the images it is altered using random transformations and rotations to create a semi-realistic dataset as in the daily life multiple times images will be found rotated, tilted, sheared etc.

The labels are adjusted from 99 labels for each age to 4 labels belonging to the following age groups:

0. Ages 0-25
1. Ages 26-50
2. Ages 51-75
3. Ages 76-100

The images are then loaded into 3 different models which are:

- [x] Simple CNN model consisting of an input layer, 2 sets of Conv2D + MaxPooling2D layers and at last the outputs from the Convolutional layers are flattened and then sent to the Dense layer consisting of 4 neurons and softmax activation for final output of the model.

- [ ] A ResNet model used through transfer learning by utilising the transfer learning module available on tensorflow. The ResNet model is applied ont the dataset consisting of images of different faces belonging to different age groups.

- [ ] A VGG16 model also used through transfer learning from tensorflow and applied to the same dataset.

The results from the model training are then evaluated using a confusion matrix and then tested with different images downloaded from the web whether they are being correctly classified or not.

### Results

- The Simple CNN model achieves an accuracy of 59% correctly classifying most images in the 0-25 age group as the dataset has maximum images in the same age group.

### Conclusion

The project was created to learn more about CNN models as well as tinker with them and the transfer learning module available on tensorflow and then using the models available through the module on different datasets.
