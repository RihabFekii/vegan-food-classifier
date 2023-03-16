import os 
import streamlit as st 
from PIL import Image
from roboflow import Roboflow
from dotenv import load_dotenv
import numpy as np
import cv2


st.set_page_config(layout="wide",
                   page_title="Vegan food finder",
                   page_icon="🌱")

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
workspace, project_name, version = ("rihab-bzsjy", "vegan-food-finder", 2)

if  "confidence_threshold" not in st.session_state:
    confidence_threshold = 40 
if  "overlap_threshold" not in st.session_state:
    confidence_threshold = 30


#########################
###### App logic 
#########################
def draw_bboxes(uploaded_img, json_predictions):

    img = cv2.imread(uploaded_img)

    for bounding_box in json_predictions['predictions']:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=1)
        
        cv2.putText(
            img,
            bounding_box["class"],
            (int(x0), int(y0) - 10),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.6,
            color = (255, 255, 255),
            thickness=2
        )
    
    cv2.imwrite("image_bounding_boxes.jpg", img)

def run_inference(uploaded_img):
    rf = Roboflow(api_key=API_TOKEN)
    project = rf.workspace(workspace).project(project_name)
    model = project.version(version).model

    prediction = model.predict(uploaded_img,
                  confidence= int(st.session_state.confidence_threshold),
                  overlap=int(st.session_state.overlap_threshold))

    return prediction

st.title("Find me vegan food 👀")

st.subheader("Let AI figure out if your dish is vegan 🌱 or not 🍕")

# ask user to upload an image
uploaded_image = st.file_uploader("Take a picture of your meal",
                                  type=["png", "jpg", "jpeg"],
                                  accept_multiple_files=False)


if uploaded_image is not None:
    col1, col2 = st.columns(2)
    
    # read image with PIL 
    image = Image.open(uploaded_image)
    # convert image to be able to do inference with Roboflow 
    rf_img = np.array(image)

    # display image
    col1.image(image, caption="Uploaded Image")

    ## get the confidence and ovelap values needed for the inference 
    confidence_threshold = col2.slider("Confidence threshold (%):", 0, 100, 40, 1,
                                        key="confidence_threshold",
                                        help="What is the minimum acceptable confidence level for displaying a bounding box?"
                                        )
    overlap_threshold = col2.slider("Overlap threshold (%):", 0, 100, 30, 1,
                                    key="overlap_threshold",
                                    help="What is the maximum amount of overlap permitted between visible bounding boxes?"
                                    )

    predict = col2.button("Is this dish vegan?")

    if predict: 
        prediction = run_inference(rf_img)
        json_prediction = prediction.json()
        st.write(json_prediction)




    
    