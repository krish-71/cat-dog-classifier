# 🐶🐱 Cat vs Dog Image Classifier

A deep learning project using **TensorFlow**, **Streamlit**, and **MobileNet** that classifies uploaded images as either **cats** or **dogs**.

---

## 🚀 Features

- ✅ Upload image from UI
- ✅ Real-time prediction with **confidence score**
- ✅ Preprocessing visualization
- ✅ Confusion matrix + Misclassified images
- ✅ Option to retrain with new images
- ✅ Stylish Streamlit UI

---

## 📁 Project Structure

cat_dog_classifier/
├── app.py # Streamlit web app
├── model.py # Model architecture (MobileNet)
├── train.py # Model training pipeline
├── preprocess.py # Image preprocessing utils
├── requirements.txt # Dependencies
├── README.md # You're here
├── ICP_model.h5 # Trained model (after training)
└── dataset/
├── cats/
└── dogs/

---

## 🧪 How to Run

### 🔧 Install Requirements

pip install -r requirements.txt

🚀 Start the App

streamlit run app.py

📂 Dataset

Use the classic Dogs vs Cats dataset (1000 images per class is enough).

Folder structure:

dataset/
├── cats/
└── dogs/

📊 Sample Output

Prediction result with confidence

Confusion matrix

Misclassified image grid

✨ Credits

Project built as part of an end-to-end AI learning guide covering:

CNNs

Transfer learning

Data preprocessing

Streamlit deployment

🌐 Live Demo (optional)

(You can host it on Streamlit Cloud — let me know if you want help.)
