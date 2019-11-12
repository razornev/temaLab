from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.losses import categorical_crossentropy
import cv2
import numpy as np


# 261 mercedes 0-260 idx
# 168 toyota 0-169 idx
# 131 volvo 0-130 idx

mercNum = 10
toyNum = 10
volvNum = 10

fileNames = []

brandLengths = {"mercedes":mercNum, "toyota":toyNum, "volvo":volvNum}

fileNamesIndex = 0
for brand in brandLengths:
    for i in range(0, brandLengths[brand]):
        fileNames.insert(fileNamesIndex, "padded" + brand + "-" + str(i) + ".jpg")
        fileNamesIndex += 1

np.random.shuffle(fileNames)

x = []
y = fileNames

maxHeight = 0
maxWidth = 0


for fileName in y:
    if("volvo" in fileName): brand = "volvo"
    elif("mercedes" in fileName): brand = "mercedes"
    else: brand = "toyota"
    index = y.index(fileName)
    origFileName = fileName
    if brand is 'volvo':
        y[index] = (1, 0, 0)
    elif brand is 'mercedes':
        y[index] = (0, 1, 0)
    else:
        y[index] = (0, 0, 1)
    #x.append(cv2.cvtColor(cv2.imread("./reducedTrainSet/"+brand+"/"+origFileName), cv2.COLOR_BGR2GRAY))
    x.append(cv2.imread("./reducedTrainSet/" + brand + "/" + origFileName))

x = np.array(x)
y = np.array(y)


model = Sequential([
    Conv2D(8, (100, 100), strides=8, padding="valid", input_shape=(2122, 4209, 3), activation='relu'),
    MaxPool2D(pool_size=(3, 3)),
    Flatten(),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile("adam", categorical_crossentropy, [categorical_crossentropy])

for i in range(0, 10):
    model.fit(x, y, batch_size=1, epochs=10, verbose=1)

print(y)
print(model.predict(x))
