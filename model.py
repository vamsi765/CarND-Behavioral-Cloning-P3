import csv
import cv2
import numpy as np
import sklearn
import random

lines = []
with open('data/driving_log_truncated.csv') as csvfile:
    reader = csv.reader(csvfile)
    skip_first_line = True
    for line in reader:
        if skip_first_line:
            skip_first_line=False
            continue
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = (train_test_split(lines, test_size=0.2))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

                if random.random()>0.5:
                    center_image = cv2.flip(center_image,1)
                    center_angle = -center_angle
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

BATCH_SIZE=32
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Activation, MaxPooling2D, Dropout, Lambda, Conv2D


def lenet_2_model():
  model = Sequential()
  
  model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160, 320, 3)))
  #model.add(Lambda(lambda x: x/255.0 - 0.5))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  
  model.add(Dense(1))
  return model

def lenet_model():
  model = Sequential()
  
  model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160, 320, 3)))
  #model.add(Lambda(lambda x: x/255 - .5))
  model.add(Conv2D(32, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  
  model.add(Conv2D(50, 5, 5))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  
  model.add(Flatten())
  model.add(Dense(500))
  model.add(Activation('relu'))
  
  model.add(Dropout(0.5))
  
  model.add(Dense(1))
  return model
  

def nvidia_model():
    input_shape = (160,320,3)
    model = Sequential()
    #model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = input_shape, output_shape = input_shape))
    model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=input_shape))
    
    model.add(Conv2D(3,3,3, subsample=(2,2), border_mode='same',activation='elu'))
    model.add(Conv2D(6,5,5, subsample=(2,2), border_mode='same',activation='elu'))
    model.add(Conv2D(9,5,5, subsample=(2,2), border_mode='same',activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(12,3,3, border_mode='same',activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model

#model=lenet_2_model();
#model=lenet_model();
model=nvidia_model();

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit_generator(train_generator,
        steps_per_epoch=len(train_samples)/BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=len(validation_samples)/BATCH_SIZE,
        shuffle=True, epochs=107, use_multiprocessing=True)

model.save("model.h5")

#steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
