# Dog Breed Classifier

## Project Overview

This project is to show the power of transfer learning and creating an API on Google cloud platform. The model is uses the trained layers from Inception Resnet V2 and appending addition convolutional layers to narrow down the classes probability distribution. The new layers are trained on the Kaggle Dog breed dataset which has 120 dog breeds and the model achieves a test accuracy of 98%. To properly demonstrate the AI model I have created a sample react website and react native app.
### Table of Contents

- [Dataset](#dataset)
- [Project Phases](#project-phases)
  - [Phase 0 - Research](#phase-0---research)
  - [Phase 1 - Data Collection and Pre-processing](#phase-1---data-collection-and-pre-processing)
  - [Phase 2 - Model](#phase-2---model)
  - [Phase 3 - Model Evaluation and Refinement](#phase-3---model-evaluation-and-refinement)
  - [Phase 4 - API Endpoint](#phase-5---API-Endpoint)
  - [Phase 5 - React Website](#phase-5---React-Website)
  - [Phase 6 - Mobile App](#phase-6---Mobile-App)
  - [Phase 7 - Further Work](#phase-7---further-work)
- [Languages](#Languages)
- [Packages](#Packages)

## Dataset

The dataset used in this project can be found [here](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery). For more information on the dataset. This specific dataset was chosen because of how "clean" the data was. There are a lot of classes and Kaggle is a very trusted site for accurate datasets.  Therefore, I made the choice to download and utilise these images to maximum via data augmentations and fine-tuning the model for the specific image dimensions.
## Project Phases

### Phase 0 - Research

- Acquire and understand the data.
- Define the task.
- Get acquainted with the existing tools and technologies that are available.

### Phase 1 - Data Collection and Pre-processing

- Download the dataset.
- Explore the dataset.
- Pre-process the data.

### Phase 2 - Model

- Implement a simple CNN architecture with a few convolutional and connected layers.
- Train the CNN on the provided dataset.
### Phase 3 - Model Evaluation and Refinement

The model follows a typical CNN structure:
Input --> Convolutional Layers --> Pooling Layers --> Fully Connected Layers

- Try multiple resizing dimension
- Apply different probabilities when augmenting images

### Phase 4 - API Endpoint

- Create a Google API folder
- Upload the model and necessary assets
- Test connections and prepare data accordingly
### Phase 5 - React Website

- set up a drag and drop image zone
- Use react UseStates and UseEffect functions to account for changes
- Communicate with the API
### Phase 6 - Mobile App

- Translate Website into mobile app
- Get camera permissions and upload the photo to Google Cloud API
### Phase 7 - Further Work

- Speed up model processing 
- Have the model installed on the mobile itself
- Reduce size of the TF model
## Languages

- Python
- Jupyter Notebook
- Javascript
## Packages

Main packages:
- Numpy
- Pandas
- Tensorflow


