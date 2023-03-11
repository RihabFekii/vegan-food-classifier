from roboflow import Roboflow

# conect to roboflow 
rf = Roboflow(api_key="Yw9XY1DSYpAHtFUrSCUT")

# get project 
project = rf.workspace("rihab-bzsjy").project("vegan-food-finder")

# download dataset locally in a custom format for yolov5
dataset = project.version(2).download("yolov5")

