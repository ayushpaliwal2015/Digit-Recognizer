## Create Virtaul Env       ## python -m venv /path/to/mnist_web_venv
## Activate Virtual Env     ## path/to/mnist_web_venv/Scripts/activate    ## C:/users/ayush/mnist_web_venv/Scripts/activate
## Upgrade pip              ## pip install --upgrade pip
## Install dependencies     ## pip install -r requirements.txt
## If Python isn't on your Windows path, you may need to type out the full path to pyinstaller to get it to run.
## AWS EC2: ssh -i "ec2_key_dig_rec.pem" ubuntu@ec2-34-201-251-31.compute-1.amazonaws.com
## 

from flask import Flask, render_template, request
import numpy as np
import base64
import cv2
import tensorflow as tf

from model.train import TrainCNN
from helpers.functions import read_config

# Load config 
config = read_config().get("PARAMETERS")

# Train or load CNN model
CNN = TrainCNN(config)
cnn_model = CNN.load_or_train_model()

# Initialize flask app
app = Flask(__name__)

# Handle GET request
@app.route('/', methods=['GET'])
def drawing():
    global config 
    html_template_path = config.get("html_template_path")
    return render_template(html_template_path)

# Handle POST request
@app.route('/', methods=['POST'])
def canvas(model = cnn_model):
    global config 
    html_template_path = config.get("html_template_path")
    img_cols = int(config.get("img_cols"))
    img_rows = int(config.get("img_rows"))

    # Recieve base64 data from the user form
    canvasdata = request.form['canvasimg']
    encoded_data = request.form['canvasimg'].split(',')[1]

    # Decode base64 image to python array
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert 3 channel image (RGB) to 1 channel image (GRAY)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to (28, 28)
    gray_image = cv2.resize(gray_image, (img_rows, img_cols), interpolation=cv2.INTER_LINEAR)

    # Expand to numpy array dimenstion to (28, 28, 1)
    img = np.expand_dims(gray_image, axis = 0)

    # Scale image 
    img = tf.constant(img)/255

    try:
        prediction = np.argmax(model.predict(img))
        return render_template(html_template_path, response=str(prediction), canvasdata=canvasdata, success=True)
    except Exception as e:
        return render_template(html_template_path, response=str(e), canvasdata=canvasdata)

if __name__ == '__main__':
    app.run()
