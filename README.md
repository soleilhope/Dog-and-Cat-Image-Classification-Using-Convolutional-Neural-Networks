# Dog-and-Cat-Image-Classification-Using-Convolutional-Neural-Networks

Hi!
This here is my final project for Machine Learning - CSC 4850. My group members and I chose this project because we all shared the same interest in learning more about neural networks and wanted to understand how image classifcation works in detail. This GitHub repository contains an implementation of a deep learning model for classifying images of dogs and cats. 
## Team Members
* Soleil St Louis: Assisted in building the GradCam model and Saliency Map
* Muhammand Khan: Assisted in training the model and figuring out bugs in the program.
* Prahalad Muralidharam: Ensure smooth collaboration in the project and brought the idea of this project
* Calvin Tran: Implemented all the models and visualization of the models(GradCam, GradCam++, Saliency Map), fine-tuned the hyperparameters for optimal performance, coordinated the team's efforts.
## Objectives
* Build a CNN network to classify Dogs and Cats using a pre-trained model(VGG-16)
* Build a KNN model to classify Dogs and Cats 
* Be able to visualize the features for each model using the Grad-Cam, Grad-Cam++, and Saliency Map
* Be able to use cross-validation techniques to validate our models

## DataSet Overview
* The dataset is taken from [kaggle](https://www.kaggle.com/datasets/chetankv/dogs-cats-images) and contains images of cats and dogs.
* The dataset was pre-separated into cats and dogs.
* The dataset had a 20% testing set and 80% training set

## Features
* Image classification: The model is trained to classify images as dogs or cats, leveraging convolutional neural networks (CNNs) for high-performance image recognition.
* Deep learning framework: The implementation is based on many popular deep learning frameworks such as Tensorflow and Keras
* Preprocessing: The repository includes scripts for preprocessing the image dataset, including resizing, normalization, and data augmentation techniques, to enhance the model's ability to generalize well to new images.
* Model architecture: The deep learning model is built using a combination of convolutional and pooling layers, followed by fully connected layers for classification. The architecture is optimized for achieving high accuracy in differentiating between dogs and cats.
* Training and evaluation: The code provides functionality for training the model on a labeled dataset of dog and cat images, as well as evaluating its performance using various metrics such as accuracy, precision, and recall.
* Jupyter notebooks: The repository contains Jupyter notebooks that walk through the process of training the model, visualizing the results, and experimenting with different hyperparameters and techniques to improve performance.
## Final Report

[Final Report](Final Report Dog and Cat Image Classification Using Convolutional Neural Networks(CNN).pdf)

## License

[MIT](https://choosealicense.com/licenses/mit/)
