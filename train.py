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

# download dataset locally in a custom format for yolov5
dataset = project.version(2).download("yolov5")
