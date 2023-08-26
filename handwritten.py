#Import all the required Libraries
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from functools import partial
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

%matplotlib inline
np.random.seed(1)
#Importing the training and testing data
training_set = pd.read_csv("../input/digit-recognizer/train.csv")
testing_set = pd.read_csv("../input/digit-recognizer/test.csv")

print("The shape of training data - ", training_set.shape)
print("The shape of testing data - ", testing_set.shape)
#Let's split the label and the pixel values from the training data
X_train = training_set.drop(['label'], axis=1).values.reshape(training_set.shape[0], 28, 28, 1)
y_train = training_set['label'].copy()
testing_set = testing_set.values.reshape(testing_set.shape[0], 28, 28, 1)

print("The shape of X_train is : ", X_train.shape)
print("The shape of y_train is : ", y_train.shape)
print("The shape of testing_set is : ", testing_set.shape)
#It is important to have a validation set to assess the model's performance.
#We will be creating a validation set which contains atleast 6,800 images.

#first shuffling the dataset
shuffled_indices = np.random.permutation(42000)
X_train = X_train[shuffled_indices]
y_train = y_train[shuffled_indices]

#Creating a validation set from the last 6,800 images
X_train, X_valid = X_train[:35200], X_train[35200:]
y_train, y_valid = y_train[:35200], y_train[35200:]

print("The shape of X_train is : ", X_train.shape)
print("The shape of y_train is : ", y_train.shape)
print("The shape of X_valid is : ", X_valid.shape)
print("The shape of y_valid is : ", y_valid.shape)
#Let's see how many images are there per class(digit)

print(y_train.value_counts())

#Each class has around 3200 to 3900 images. So the chance of class imbalance is less
#Create a basic Convolutional Neural Network model

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding='same')

Digit_Recognizer = Sequential([
                        DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
                        keras.layers.MaxPooling2D(pool_size=2),
                        DefaultConv2D(filters=128),
                        DefaultConv2D(filters=128),
                        keras.layers.MaxPooling2D(pool_size=2),
                        DefaultConv2D(filters=256),
                        DefaultConv2D(filters=256),
                        keras.layers.MaxPooling2D(pool_size=2),
                        keras.layers.Flatten(),
                        Dense(units=128, activation='relu'),
                        Dropout(0.5),
                        Dense(units=64, activation='relu'),
                        Dropout(0.5),
                        Dense(units=10, activation='softmax'),
                   ])

Digit_Recognizer.compile(optimizer="nadam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

#Summarizing the model
Digit_Recognizer.summary()
#Training the model
filepath = "./model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]
history = Digit_Recognizer.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_valid, y_valid), 
                     verbose=1, callbacks=callbacks_list)

#testing on an image
Digit_Recognizer = load_model('./model.h5')     #Loading the best model
plt.imshow(X_valid[1234].reshape(28, 28))
print("The digit recognized by the model is : ", Digit_Recognizer.predict_classes(X_valid[1234].reshape(1, 28, 28, 1)))
plt.axis('off')
plt.show()

#The model predicts with almost 99.2% validation accuracy
test_predictions = pd.DataFrame(Digit_Recognizer.predict_classes(testing_set), index=range(1, 28001))
test_predictions.to_csv('./Submissions.csv', index_label=['ImageId', 'Label'])


