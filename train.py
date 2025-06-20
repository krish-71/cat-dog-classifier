# Phase 4: Train the Model
# train.py
from model import create_model
from preprocess import get_data_generators
import pickle

def train_and_save_model():
    model = create_model()
    train_data, val_data = get_data_generators()

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )
    model.save("ICP_model.h5")

    with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    return model, history


if __name__ == "__main__":
    train_and_save_model()
