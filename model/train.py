import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json 
import os
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from helpers.functions import load_csv_data
import numpy as np

class TrainCNN:
    # Collect the model training hyperparameters 
    def __init__(self, config):
        self.config = config
        self.n_class = 10

    # Load external data from csv
    def load_external_data(self):
        (self.x_train, self.y_train) = load_csv_data(self.config.get("train_data_path"))
        (self.x_test, self.y_test)   = load_csv_data(self.config.get("test_data_path"))

    # Load Keras data
    def load_keras_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    # Loading the data
    def load_data(self):
        use_external_data = bool(int(self.config.get("use_external_data")))
        if use_external_data:
            self.load_external_data()
        else:
            self.load_keras_data()

    # Preparing the data
    def process_data(self):
        img_cols = int(self.config.get("img_cols"))
        img_rows = int(self.config.get("img_rows"))

        # Reshape data to have a single channel 
        self.x_train = self.x_train.reshape(-1, img_rows, img_cols, 1)
        self.x_test = self.x_test.reshape(-1, img_rows, img_cols, 1)
        self.input_shape = (img_rows, img_cols, 1)

        # Convert to float type
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')

        # Normalize the values
        self.x_train /= 255
        self.x_test /= 255

        # Convert Labels to categorical type
        self.y_train = keras.utils.np_utils.to_categorical(self.y_train, self.n_class)
        self.y_test = keras.utils.np_utils.to_categorical(self.y_test, self.n_class)

    # Data augmentation to prevent overfitting
    def augment_data(self):
        batch_size = int(self.config.get("batch_size"))
        rotation_range = int(self.config.get("rotation_range"))
        zoom_range = self.config.get("zoom_range")
        zoom_range = list(map(float, zoom_range.split(",")))
        width_shift_range = float(self.config.get("width_shift_range"))
        height_shift_range = float(self.config.get("height_shift_range"))

        data_gen = ImageDataGenerator(
                featurewise_center = False,             # set input mean to 0 over the dataset
                samplewise_center = False,              # set each sample mean to 0
                featurewise_std_normalization = False,  # divide inputs by std of the dataset
                samplewise_std_normalization = False,   # divide each input by its std
                zca_whitening = False,                  # apply ZCA whitening
                rotation_range = rotation_range,        # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = zoom_range,                # Randomly zoom image 
                width_shift_range = width_shift_range,  # randomly shift images horizontally (fraction of total width)
                height_shift_range = height_shift_range,# randomly shift images vertically (fraction of total height)
                horizontal_flip = False,                # randomly flip images
                vertical_flip = False)                  # randomly flip images

        self.train_gen = data_gen.flow(self.x_train, self.y_train, batch_size = batch_size)
        self.test_gen = data_gen.flow(self.x_test, self.y_test, batch_size = batch_size)

    # Creating the Model Architecture
    def create_cnn_model(self):

        self.model = Sequential()
        self.model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = self.input_shape))
        self.model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
        self.model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dense(self.n_class, activation = 'softmax'))
        self.model.compile(loss = keras.losses.categorical_crossentropy, optimizer = tf.keras.optimizers.Adam(), metrics = ['accuracy'])
        self.model.summary()

    # Training the Model
    def train_cnn_model(self):
        batch_size = int(self.config.get("batch_size"))
        epochs = int(self.config.get("epochs"))

        train_steps = self.x_train.shape[0] // batch_size
        valid_steps = self.x_test.shape[0] // batch_size

        es = keras.callbacks.EarlyStopping(
                monitor = "accuracy",           # metrics to monitor
                patience = 3,                   # how many epochs before stop training
                verbose = 1,                    # provide update messages
                mode = "max",                   # to stop training if accuracy does not improve
                restore_best_weights = True,    # restore best weights after training
            )

        rp = keras.callbacks.ReduceLROnPlateau(
                monitor = "accuracy",           # metrics to monitor
                factor = 0.2,                   # lr shrinkage factor
                patience = 2,                   # lr reducing epoch patience
                verbose = 1,                    # provide update messages
                mode = "max",                   # to reduce lr if accuracy does not improve
                min_lr = 0.00001,               # lower limit of lr
            )

        self.model.fit(self.train_gen, 
                              epochs = epochs, 
                              steps_per_epoch = train_steps,
                              validation_data = self.test_gen,
                              validation_steps = valid_steps, 
                              verbose = 1,
                              callbacks=[es, rp])


    # Evaluating the Predictions on the Test data
    def evaluate_cnn_model(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose = 0)

    # Saving the model for Future Inferences
    def save_cnn_model(self):
        json_model_path = self.config.get("json_model_path")
        h5_model_path = self.config.get("h5_model_path")

        model_json = self.model.to_json()
        with open(json_model_path, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(h5_model_path)

    # Open and load model files 
    def load_saved_model(self):
        # opening and store file in a variable
        json_model_path = self.config.get("json_model_path")
        json_file = open(json_model_path,'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # use Keras model_from_json to make a loaded model
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        h5_model_path = self.config.get("h5_model_path")
        loaded_model.load_weights(h5_model_path)

        # compile and evaluate loaded model
        loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        self.model = loaded_model

    # Delete model dump files if exist
    def remove_model_files(self):
        try:
            os.remove(self.config.get("h5_model_path"))
        except OSError:
            pass

        try:
            os.remove(self.config.get("json_model_path"))
        except OSError:
            pass

    # Train a new model and save it
    def train_model(self):
        self.load_data()
        self.process_data()
        self.augment_data()
        self.create_cnn_model()
        self.train_cnn_model()
        self.evaluate_cnn_model()
        self.save_cnn_model()

    # Load a model if exists else train a new one
    def load_or_train_model(self):

        if os.path.exists(self.config.get("h5_model_path")) and os.path.exists(self.config.get("json_model_path")):
            self.load_saved_model()
        else:
            self.remove_model_files()
            self.train_model()
                
        return self.model
