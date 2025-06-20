# Phase 3: Build CNN Model
# model.py
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#def create_model():
    #model = Sequential([
        #Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        #MaxPooling2D(2,2),
        #Conv2D(64, (3,3), activation='relu'),
        #MaxPooling2D(2,2),
        #Flatten(),
        #Dense(128, activation='relu'),
        #Dropout(0.5),
        #Dense(1, activation='sigmoid')
    #])

    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #return model
# model.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam

def create_model():
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base for speed

    inputs = Input(shape=(128, 128, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
