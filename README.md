# Find me vegan food 👀 
A Computer Vision project for vegan food 🌱 recognition. 

A web-based app to classify vegan food from non-vegan food using Roboflow. 

### A step-by-step guide to this project 
1. Data annotation using Roboflow's annotation tool
2. Data pre-processing
3. Training an ML model on Roboflow
4. Deploying the ML model
5. Creating a web app with Streamlit 


# Data preparation 

You can have a nice AI project idea in mind 🚀, but oftentimes, you don't find the fitting dataset. 

It was the case in my project. What I did is, I collected some images and used a tool like Roboflow to annotate and pre-process my dataset.

*With Roboflow, not finding a dataset will no longer be an obstacle!*

## Upload data 

To create a dataset, I am using [Roboflow](https://app.roboflow.com/) to upload my raw data. 

Start by creating a **workspace**, then create a **project** and select the Computer Vision **project type** that fits your use case, as shown in the figure below: 

![create_project](/docs/create-project.png)

Upload your data via the **upload** interface. Adding **tags** to your data is highly recommended, as it enables easier search. 

![upload_data](/docs/uploaded-data.png)

## Annotate data 

The next step after uploading data on vegan and omnivore food is annotation. 

Roboflow has a built-in **annotation tool** that I will use to annotate my data for my object detection task. 

The picture below shows the annotation tool interface. I can easily draw the **bounding box** around the plate and assign the **labels** "vegan" or "omnivore" to my plate(s) in my images. 

![annotate](/docs/annotate.png)

## Create a dataset 

Once you annotate the data, the next step is to create a dataset to train the ML model on. 

You have the option to split the data within your dataset in different ways. In my case, I chose to split the data to train(87%), test(5%), and validation(8%) sets. 

## data pre-processing 

Roboflow enables you to easily make pre-processing e.g data resize, orientation, color conversion, etc. via the **Generate** interface, as shown below: 

![pre-processing](/docs/pre-processing.png)

## Data augmentation 

Since my dataset is too small with 113 images, doing data augmentation can enhance the model training performance.

The image below represents the different options available via Roboflow for data augmentation: 

![data-augmentation](/docs/augmentation.png)

I applied 90° rotation, Exposure (25%), and mosaic on my dataset. 

Finally, this is what the configuration of the newly generated version of my dataset looks like: 

![dataset-generate](/docs/dataset-config.png)