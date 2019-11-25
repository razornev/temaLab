from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from DataGenerator import DataGenerator
import cv2
import numpy as np


# 261 mercedes 0-260 idx    train:240   test:21
# 168 toyota 0-169 idx      train:150   test:18
# 131 volvo 0-130 idx       train:120   test:11
# 560 - 50  = 510

model = Sequential([
    Conv2D(8, (100, 100), strides=8, padding="valid", input_shape=(2122, 4209, 3), activation='relu'),
    Conv2D(8, (30, 30), strides=8, padding="valid", activation='relu'),
    MaxPool2D(pool_size=(3, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(30, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile("adam", categorical_crossentropy, [categorical_crossentropy])

labels = np.load("./data/validationDictionary.npy", allow_pickle=True).item()

print(labels)

list_IDs = np.arange(0, labels.__len__(), 1)

'''for i in range(0, len(list_IDs)) :
    list_IDs[i] = str(list_IDs[i])
'''

model.fit_generator(DataGenerator(list_IDs, labels, n_classes=3), epochs=10)


'''
sources:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly#data-generator
'''
