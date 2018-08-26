import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pathlib
import itertools
import os
import shutil
import fire
import mlflow
import keras
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers


class CatsDogs(object):
    def __init__(self):
        self.base_dir, self.train_dir, self.validation_dir, self.test_dir = self.set_directories()
        self.model = self.create_model()

    def set_directories(self):
        base_dir = pathlib.Path('/data')
        train_dir = base_dir/'train'
        validation_dir = base_dir/'validation'
        test_dir = base_dir/'test'
        train_cats_dir = train_dir/'cats'
        train_dogs_dir = train_dir/'dogs'
        validation_cats_dir = validation_dir/'cats'
        validation_dogs_dir = validation_dir/'dogs'
        return base_dir, train_dir, validation_dir, test_dir

    def create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(150, 150, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        print(model.summary())
        return model

    def fit(self, epochs=10, lr=1e-4, bs=20):
        self.model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.Adam(lr=lr),
                      metrics=['accuracy']
        )
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            self.base_dir,
            target_size=(150, 150),
            batch_size=bs,
            class_mode='binary'
        )
        # validation_datagen = ImageDataGenerator(rescale=1./255)
        # validation_generator = validation_datagen.flow_from_directory(
            # self.validation_dir,
            # target_size=(150, 150),
            # batch_size=bs,
            # class_mode='binary'
        # )

        history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=1600/bs,
            epochs=epochs,
            # validation_data=validation_generator,
            # validation_steps=400/bs
        )
        # score = self.model.evaluate_generator(validation_generator)
        # self.model.save(f'/output/cats_and_dogs_{epochs}_{score[1]}.h5')
        self.model.save(f'/output/cats_and_dogs_{epochs}.h5')

        # mlflow.log_param("learning_rate", lr)
        # mlflow.log_param("epochs", epochs)
        # mlflow.log_param("batch_size", bs)
        # mlflow.log_metric("val_loss", score[0])
        # mlflow.log_metric("val_accuracy", score[1])

    def predict(self, model_path, bs=20):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(150, 150),
            batch_size=bs,
            class_mode='binary'
        )
        model = keras.models.load_model(model_path)
        preds = model.predict_generator(test_generator)
        print(preds)
        submission = pd.read_csv('sampleSubmission.csv')
        submission['label'] = preds
        submission.to_csv('submission.csv', index=False)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact('submission.csv')


if __name__ == "__main__":
    fire.Fire(CatsDogs)
