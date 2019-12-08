from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, LeakyReLU, Dropout, AvgPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD, RMSprop
import pickle

input_shape = 500

model = Sequential([
    Conv2D(64, (3, 3), padding="same", input_shape=(input_shape, input_shape, 3), activation="relu"),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), padding="same", activation="relu"),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), padding="same", activation="relu"),
    Conv2D(256, (3, 3), padding="same", activation="relu"),
    Conv2D(256, (3, 3), padding="same", activation="relu"),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(512, (3, 3), padding="same", activation="relu"),
    Conv2D(512, (3, 3), padding="same", activation="relu"),
    Conv2D(512, (3, 3), padding="same", activation="relu"),
    MaxPool2D(pool_size=(2, 2)),

    Flatten(),

    Dense(512, activation="relu"),

    Dense(512, activation="relu"),
    Dropout(0.2),

    Dense(3, activation="softmax")
])

model.compile(RMSprop(lr=1e-5), categorical_crossentropy, ["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen = train_datagen.flow_from_directory(
    'data/train',
    target_size=(input_shape, input_shape),
    batch_size=2,
    class_mode='categorical')

validation_datagen = ImageDataGenerator(
    rescale=1. / 255
)

validation_datagen = validation_datagen.flow_from_directory(
    'data/newValid',
    target_size=(input_shape, input_shape),
    batch_size=2,
    class_mode='categorical')

for i in range(0, 100):
    mc = ModelCheckpoint('best_model_' + str(i) + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    history = model.fit_generator(train_datagen, validation_data=validation_datagen, epochs=2, callbacks=[mc])

    model_json = model.to_json()
    with open("./content/networks/vmmrdb_lger_ep" + str(i * 2) + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./content/networks/vmmrdb_lger_ep" + str(i * 2) + ".h5")
    # print("Saved model to disk")

    with open('/trainHistoryDict' + str(i), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
