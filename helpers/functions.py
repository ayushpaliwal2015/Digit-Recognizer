import configparser
import numpy as np

config_file_path = "./config.ini"

# function to read config file 
def read_config():
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return dict(config)


# Load external data from csv
def load_csv_data(file_path):

    data = []
    labels = []
    
    for row in open(file_path): # Openfile and start reading each row
        row = row.split(",")
        
        label = int(row[0])     
        
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        data.append(image)
        
        labels.append(label)
        
    data = np.array(data, dtype='float32')
    labels = np.array(labels, dtype="int")
    
    return (data, labels)