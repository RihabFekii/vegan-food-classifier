import logging
import os
import sys

from dotenv import load_dotenv
from roboflow import Roboflow

# parses the .env file and loads all the variables found as environment variables.
load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")

# path to one image from argument
path_image = str(sys.argv[1])
logging.info(path_image)

# conect to roboflow 
rf = Roboflow(api_key=API_TOKEN)

# get project 
project = rf.workspace("rihab-bzsjy").project("vegan-food-finder")

# retreive trained model with its version
model = project.version(2).model

# infer on a local image
print(model.predict(path_image, confidence=40, overlap=30).json())

# visualize your prediction
model.predict(path_image, confidence=40, overlap=30).save("prediction.jpg")