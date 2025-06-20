from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import glob

model_path = "ICP_model.h5"
if not os.path.exists(model_path):
    st.error("❌ Trained model not found. Please run `python train.py` first.")
    st.stop()
else:
    model = load_model(model_path)

st.title("🐶🐱 Cat vs Dog Image Classifier")

# Show random samples from dataset
st.sidebar.subheader("🔍 Data Exploration")
if st.sidebar.button("Show Random Images"):
    categories = ['cats', 'dogs']
    sample_images = []
    for cat in categories:
        folder = os.path.join("dataset", cat)
        imgs = glob.glob(folder + "/*.jpg")
        sample = random.sample(imgs, min(3, len(imgs)))
        sample_images.extend(sample)

    st.write("### 📸 Sample Images from Dataset")
    cols = st.columns(len(sample_images))
    for col, img_path in zip(cols, sample_images):
        col.image(img_path, use_column_width=True, caption=os.path.basename(img_path))

uploaded_file = st.file_uploader("Upload an image of a cat or a dog", type=["jpg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence * 100:.2f}%")

st.sidebar.subheader("📊 Train a New Model")
if st.sidebar.button("Retrain with Current Dataset"):
    from train import train_and_save_model
    with st.spinner("Training model..."):
        model, history = train_and_save_model()
        st.success("✅ Model retrained and saved!")

# Training curve
st.sidebar.subheader("📈 Training Curves")
if st.sidebar.button("Show Training Curves"):
    if os.path.exists("history.pkl"):
        with open("history.pkl", "rb") as f:
            history = pickle.load(f)

        st.line_chart({
            'Train Accuracy': history['accuracy'],
            'Validation Accuracy': history['val_accuracy']
        })
    else:
        st.warning("Training history not found. Please train the model first.")

# Confusion Matrix & Misclassified Images
st.sidebar.subheader("📉 Confusion Matrix")
if st.sidebar.button("Show Confusion Matrix"):
    from preprocess import get_data_generators
    _, val_data = get_data_generators()

    y_true = []
    y_pred = []
    misclassified_images = []

    for images, labels in val_data:
        preds = model.predict(images)
        for img, label, pred in zip(images, labels, preds):
            actual = int(label)
            predicted = int(pred > 0.5)
            y_true.append(actual)
            y_pred.append(predicted)
            if actual != predicted:
                misclassified_images.append((img, actual, predicted))

    cm = confusion_matrix(y_true, y_pred)
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Show misclassified images
    if misclassified_images:
        st.write("### ❌ Misclassified Images")
        cols = st.columns(3)
        for i, (img, actual, predicted) in enumerate(misclassified_images[:9]):
            with cols[i % 3]:
                st.image(img.numpy(), use_column_width=True, caption=f"Actual: {'Dog' if actual else 'Cat'}, Predicted: {'Dog' if predicted else 'Cat'}")
    else:
        st.write("✅ No misclassified images found!")