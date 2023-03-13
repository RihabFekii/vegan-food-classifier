# Find me vegan food 👀 
A Computer Vision project for vegan food 🌱 recognition. 

A web-based app to classify vegan food from non-vegan food using Roboflow. 

### A step-by-step guide to this project 
1. Data annotation using Roboflow's annotation tool
2. Data pre-processing
3. Training an ML model on Roboflow
4. Deploying the ML model
5. Creating a web app with Streamlit 


# Step 1: Data preparation 

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

## Data pre-processing 

Roboflow enables you to easily make pre-processing e.g data resize, orientation, color conversion, etc. via the **Generate** interface, as shown below: 

![pre-processing](/docs/pre-processing.png)

## Data augmentation 

Since my dataset is too small with 113 images, doing data augmentation can enhance the model training performance.

The image below represents the different options available via Roboflow for data augmentation: 

![data-augmentation](/docs/augmentation.png)

I applied 90° rotation, Exposure (25%), and mosaic on my dataset. 

Finally, this is what the configuration of the newly generated version of my dataset looks like: 

![dataset-generate](/docs/dataset-config.png)

# Step 2: Training 

Now that the data annotation and dataset creation and processing are completed. The next step is to train an object detection model on your dataset. Roboflow makes this process simple and possible in one click.

You can choose via the Roboflow **Versions** interface, to export your dataset to the format that fulfills your computer vision task and it will assign the fitting model(s) accordingly. 

When the training is complete, you get **notified via email**. 

## Training metrics
The great advantage is to have a real-time dashboard of the training progress. 
Since I have an object detection task, the mAP (mean average precision) is a fitting metric to track the model-trained performance. 

![training-dashboard](/docs/training.gif)

After the training is completed, you also can have an overview of the **training metrics** e.g validation set, test set, and training graphs, as shown below: 

![training-metrics](/docs/train-metrics.png)

You can also visualize the prediction of your model and the ground truth to have a better overview of how truly your model performs the task of object detection of the vegan and omnivore plates, as shown below: 

![visualize-predictions](/docs/predictions.gif)


## Dataset health check 

The best way to assess model performance besides checking the training metrics is to check the dataset class distribution, balance, annotation quality, etc. 

Thanks to the health check feature on Roboflow, this process is made straightforward, as shown below: 

![health-check](/docs/health-check.gif)

This way you can understand better your dataset and see the flaws in it that you need to correct to enhance your trained model predictions e.g correcting data unbalance by adding more data samples to the class where there are fewer samples.

## New training version 

You can generate a new version of the dataset and try out a new training with a different model. That way you can compare the metrics of the different trained models and deploy the one that gives the best outcome. 

After creating a new dataset version, I will choose a  model by clicking on the export button, as shown below: 

![new-dataset-version](/docs/new-version.png)

I will re-tarin a second version of the model using the previous experiment, as shown below: 

![train](/docs/train-with-previous.png)

If you want to do the training locally using the dataset hosted at Roboflow and when you select a model, you can copy the following code snippet that will allow you to download the data locally in a format custom to the chosen model. 

````python 
!pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="your-api-key")
project = rf.workspace("workspace-name").project("proejct-name")
dataset = project.version(2).download("yolov5")
````

# Step 3: Computer Vision model deployment 

There are multiple options for model deployment and choosing which option always depends on the use case and on how the end-user will consume the model.

## Test model 
For instance, to **test** my model I navigate to the **Deploy** tab in my project interface on Roboflow and drag and drop images or video files or use my webcam to do so, as shown below: 

![test-model](/docs/test-model.png)

## API for inference 

In the scope of this project, I want to create a web app that enables the end-user to upload a picture of a dish to know if the food is vegan or not. 

For that, creating an **API endpoint** to **trigger the inference** is suitable. 

Instead of creating your API endpoint from scratch, Roboflow hosts an API for you to trigger the inference as shown in the [inference.py](/inference.py) file. 

````python 
rf = Roboflow(api_key="your-api-token")

# get project 
project = rf.workspace("your-workspace-name").project("your-project-name")

# retreive trained model with its version
model = project.version(1).model

# infer on a local image
print(model.predict("your-image.jpeg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("your-image.jpeg", confidence=40, overlap=30).save("prediction.jpg")
````

To test the inference script, you can use the images in this folder "vegan-test.jpeg" and "omnivore-text.jpeg" and visualize the predictions of the model respectively in [prediction-vegan.jpg](/prediction-vegan.jpg) and [prediction-omnivore.jpg](/prediction-omnivore.jpg). 









