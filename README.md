# Mobility AI with urban imagery

### Using CNN to Predict Median Property Value in California



### Introduction

This project is about using the satellite and street view images to train a CNN model in order to predict the median property value in California.



### Timeline

###### Gather Information (week 1)

- Read the references papers
- Obtain the required data
- Find coding resources

###### Practice (week 2)

- Clean the data
- Apply code on the dataset

###### Review and Refine (week 3-4)

- Collect first-stage results
- Analyze the results
- Improve the model

###### Conclude (week 5)

- Collect all results
- Upload code to the GitHub repository



### Methods

##### Data Introduction

We downloaded the satellite images and street view images by using GoogleMapAPI. 

- Satellite Images (9129)

  <img src="./pic/satellite.png" alt="image-20230821035708358" style="zoom:50%;" />

- Street View Images (6440)

  <img src="./pic/streetview.png" alt="image-20230821035723975" style="zoom:50%;" />

- Census Tract (8012)

  <img src="./pic/census_tract.png" alt="image-20230821035741855" style="zoom:50%;" />

We first create and enable Google API key. Then we use geopanda to read the shapefile and extract geoid, latitude and longitude to form the coordinate file. With the API key and the coordinate file, we use GoogleMapAPI to download images. 

The name of the images: "No._FIPS" 

##### Model Introduction

###### AlexNet

It is a model designed by Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton. The model won ILSVRC(ImageNet Large Scale Visual Recognition Challenge) in 2012.

The structure of the original Alexnet:

<img src="./pic/alexnet_structure.png" alt="image-20230821033410474" style="zoom:33%;" />

In this project, we use the Alexnet pretrained on ImageNet. Also, since we want the model to predict the specific value, we change the out features of the last layer of the model from 1000 to 1.

###### VGG16

VGG stands for Visual Geometry Group; it is a standard deep Convolutional Neural Network (CNN) architecture with multiple layers. 

It was one of the most popular models submitted to ILSVRC-2014. It replaces the large kernel-sized filters with several 3×3 kernel-sized filters one after the other, thereby making significant improvements over AlexNet. (https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/)

Structure of VGG16:

![image-20230821034303517](./pic/VGG16.png)

Same as AlexNet, we use the pretrained VGG16 model in this project and modify the out feature of the last layer of the model from 1000 to 1.

##### Evaluation

We use MSE and $R^2$ to evaluate the performance of the model. 

<img src="./pic/formulation.png" alt="image-20230821034610913" style="zoom:35%;" />

- MSE = mean squared error
- R² = coefficient of determination
- RSS = sum of squares of residuals
- TSS = total sum of squares

MSE means the average squared difference between the estimated values and the actual value. 

An R-Squared value shows how well the model predicts the outcome of the dependent variable. R-Squared values range from 0 to 1. An R-Squared value of 0 means that the model explains or predicts 0% of the relationship between the dependent and independent variables.

##### Training

The hyperparameters we use when training are listed as follows:

- Dataset:
  - 70% training set, 30% test set
  - satellite images: 6390+2739
  - street view images: 4508+1932

- Loss function: MSE
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Epochs:
  - Satellite images, Alexnet: 50 epochs
  - Satellite images, VGG16: 80 epochs
  - Street view images, Alexnet: 50 epochs
  - Street view images, VGG16: 70 epochs



### Results

##### 1. MSE and $R^2$ value

![image-20230821040301715](./pic/value.png)

The model trained by satellite images can have the higher $R^2$ and lower MSE value than the model trained by street view images.

The VGG model can have the lower MSE value and slightly higher $R^2$ value than the Alexnet.

So based on the MSE and $R^2$ results, the VGG model trained by satellite images performs better than the other models.

##### 2. Scatter Plots

![image-20230821041444144](./pic/scatter_plots.png)

The scatter plots of ground truth median property value v.s. our predictions. The red line is y=x, which means the perfect predictor.

For the model trained by street view images, there are more dots fall in the red area, which means the model tends to predict lower value compared to the model trained by satellite images. The dots should be close to the red line.

Since there are 9129 FIPS in the shapefile but only 8012 data in the census tract file, for the FIPS not in the census tract file, we set the value into 0. The predictions are not equal to 0, so when the ground truth is 0, the dots form into a straight line.

##### 3. Visualization

![image-20230821042633121](./pic/visualization.png)

Predictions from model trained by satellite images are more closer to the ground truth.  



### Conclusion and Discussion

1. Model trained by satellite images performs better than model trained by street view images. We guess this is because  the street view images lack representativeness and information so one street view image per FIPS may be not enough.

2. VGG16 is slightly better than AlexNet. It is reasonable because VGG16 has deeper layers than AlexNet so this model can better extract features. Also, VGG16 has smaller kernel sizes compared to AlexNet, which can reduce computational costs. 
