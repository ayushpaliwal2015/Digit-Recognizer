import configparser
import numpy as np
import cv2
import base64
import time
import json
import os

config_file_path = "./config.ini"

# function to read config file 
def read_config():
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return dict(config)


# Load external data (user input data) from jsons
def load_external_data(folder_path):

    data = []
    labels = []
    
    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            input_data = json.load(f)

        data.append(input_data.get("img_array"))
        
        labels.append(input_data.get("img_label"))
        
    data = np.array(data, dtype='float32')
    labels = np.array(labels, dtype="int")
    
    return (data, labels)


# Parse image data from request 
def parse_image_data_from_request(request, img_rows, img_cols):

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

    return canvasdata, img


# Save image and user input
def save_user_input_data(external_data_path, img, user_input):

    time_epoch_now_ms = int(time.time()*1000)

    # Make file name out of timestamp
    file_name = str(time_epoch_now_ms) + ".json"
    file_path = external_data_path + "/" + file_name

    # Store user data in dict
    user_data = {"img_array": img.tolist()[0], "img_label": int(user_input)}

    # Save the dictionary as a JSON file
    with open(file_path, "w") as f:
        json.dump(user_data, f)

def train_test_data_split(X, y, split_fraction):

    # split the array into training and testing sets
    split_idx = int(split_fraction * X.shape[0])

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return (X_train, y_train), (X_test, y_test)
