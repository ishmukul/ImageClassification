'''
Image classification of for Kaggle cats and dogs challenge based on tutorial on Keras website.
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Working on some modifications and improvement.
'''

# importing required modules
from zipfile import ZipFile
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utilities import *

# ==========================================
# File operation: Extracting data from zip and organizing in directory structure
# First extract training data into cats and dogs folder form the training zip file
if not (os.path.isdir("data/train/cats") and os.path.isdir("data/train/dogs")):
    # specifying the zip file name
    file_name = "data/train.zip"

    # opening the zip file in READ mode
    with ZipFile(file_name, 'r') as z:
        # printing all the contents of the zip file
        z.infolist()
        for zipinfo in z.infolist():
            if zipinfo.filename[-1] == '/':
                continue
            zipinfo.filename = os.path.basename(zipinfo.filename)
            if 'cat' in zipinfo.filename:
                z.extract(zipinfo, "data/train/cats")
            elif 'dog' in zipinfo.filename:
                z.extract(zipinfo, "data/train/dogs")
else:
    print("Training folder already exist")


# ==========================================
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000  # 22778
nb_validation_samples = 800  # 2222
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# ==========================================
# Building model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

file_path = "models/best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor="val_loss",
                              verbose=1, save_best_only=True, mode="min")
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=8)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=2,
    callbacks=[check_point, early_stop]
)

# ==========================================
# Save weights for future use
model.save_weights('models/model_weights.h5')
model.save("models/model.h5")


# ====================================
# Plot history
plt.close('all')

AxisLabel = ["Epochs", "Accuracy"]
FName = 'figures/Accuracy.png'
# FName = None
plot_metric(history, metric_name='accuracy', axis_label=AxisLabel, graph_title="Accuracy plot", file_name=FName)

AxisLabel = ["Epochs", "Loss"]
FName = 'figures/Loss.png'
# FName = None
plot_metric(history, metric_name='loss', axis_label=AxisLabel, graph_title="Loss plot", file_name=FName)
