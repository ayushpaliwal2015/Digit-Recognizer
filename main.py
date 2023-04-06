## Create Virtaul Env       ## python -m venv /path/to/mnist_web_venv
## Activate Virtual Env     ## path/to/mnist_web_venv/Scripts/activate    ## C:/users/ayush/mnist_web_venv/Scripts/activate
## Upgrade pip              ## pip install --upgrade pip
## Install dependencies     ## pip install -r requirements.txt
## If Python isn't on your Windows path, you may need to type out the full path to pyinstaller to get it to run.
## AWS EC2: ssh -i "ec2_key_dig_rec.pem" ubuntu@ec2-34-201-251-31.compute-1.amazonaws.com
## 

from flask import Flask, render_template, request
import numpy as np
from model.train import TrainCNN
from helpers.functions import read_config, parse_image_data_from_request, save_user_input_data
import tensorflow as tf


# Load config 
config = read_config().get("PARAMETERS")

img_cols = int(config.get("img_cols"))
img_rows = int(config.get("img_rows"))
html_template_feedback_path = config.get("html_template_feedback_path")
html_template_path = config.get("html_template_path")
external_data_path = config.get("external_data_path")


# Train or load CNN model
CNN = TrainCNN(config)
cnn_model = CNN.load_or_train_model()


# Initialize flask app
app = Flask(__name__)


# Handle GET request With Feedback
@app.route('/feedback', methods=['GET'])
def get_drawing_feedback():
    global html_template_feedback_path 
    return render_template(html_template_feedback_path)


# Handle POST With Feedback request
@app.route('/feedback', methods=['POST'])
def post_canvas_prediction_feedback(model = cnn_model):
    global html_template_feedback_path, img_cols, img_rows, external_data_path

    # Revieve user input 
    user_input = request.form['input']

    # Parse image data from request to get an image
    canvasdata, img = parse_image_data_from_request(request, img_rows, img_cols)

    # Save image and user input ONLY if we recieve user input
    if user_input:
        save_user_input_data(external_data_path, img, user_input)

    # Scale image 
    img = tf.constant(img)/255

    try:
        prediction = np.argmax(model.predict(img))
        return render_template(html_template_feedback_path, response=str(prediction), canvasdata=canvasdata, success=True)
    except Exception as e:
        return render_template(html_template_feedback_path, response=str(e), canvasdata=canvasdata)


# Handle GET request Without Feedback
@app.route('/', methods=['GET'])
def get_drawing():
    global html_template_path 
    return render_template(html_template_path)


# Handle POST request Without Feedback
@app.route('/', methods=['POST'])
def post_canvas_prediction(model = cnn_model):
    global html_template_path, img_cols, img_rows

    # Parse image data from request to get an image
    canvasdata, img = parse_image_data_from_request(request, img_rows, img_cols)

    # Scale image 
    img = tf.constant(img)/255

    try:
        prediction = np.argmax(model.predict(img))
        return render_template(html_template_path, response=str(prediction), canvasdata=canvasdata, success=True)
    except Exception as e:
        return render_template(html_template_path, response=str(e), canvasdata=canvasdata)


if __name__ == '__main__':
    app.run()
