import os

from roboflow import Roboflow
from dotenv import load_dotenv


# parses the .env file and loads all the variables found as environment variables.
load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")

# conect to roboflow 
rf = Roboflow(api_key=API_TOKEN)

# get project 
project = rf.workspace("rihab-bzsjy").project("vegan-food-finder")

# retreive trained model with its version
model = project.version(2).model

# infer on a local image
print(model.predict("your-image.jpeg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("your-image.jpeg", confidence=40, overlap=30).save("prediction.jpg")