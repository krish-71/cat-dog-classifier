# Phase 2: Image Preprocessing
# preprocess.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_data = train_gen.flow_from_directory(
        'dataset',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_data = train_gen.flow_from_directory(
        'dataset',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_data, val_data