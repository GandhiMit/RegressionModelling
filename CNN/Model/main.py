import pandas as pd

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


def model(image_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=image_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def function(test, train):
    # test  dataset
    train_data_dir = train
    test_data_dir = test

    #  Hyperparameter Tuning
    epochs = 10
    batch_size = 12
    image_width, image_height = 150, 150
    input_shape = (image_width, image_height, 3)
    # Model
    CNN_model = model(input_shape)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary'
    )
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    CNN_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size
    )

    CNN_model.save('cat_dog_classification.h5')


if __name__ == "__main__":
    dir_test = "../Data_set/test"
    dir_train = "../Data_set/train"
    function(dir_test, dir_train)
