#  🐱🐶 Image-Classification-of-Cats-and-Dogs-using-CNN-Project
This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs. CNNs are a class of deep learning models specifically designed for analyzing visual data. Using the popular Cats vs. Dogs dataset, the model is trained to identify whether an input image contains a cat or a dog with high accuracy.
Also, this project demonstrates how to build and train a **Convolutional Neural Network (CNN)** model to classify images of cats and dogs using **TensorFlow** and **Keras**.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📁 Dataset
- **Source**: [Kaggle - Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data)
- Contains thousands of labeled images of cats and dogs.
- Images are resized to **256x256** for model input.

> ⚠️ **Note:**  
> To access the dataset, you will need a **Kaggle account** and an **API key**.  
> Follow the instructions below to set it up.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🔐 Setting up Kaggle API Key

1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Scroll to **"API" section** → Click on **"Create New API Token"**
3. A file named kaggle.json will be downloaded.
4. Place it in your working directory or upload it in Google Colab.
5. Run this code to set up:

python
!pip install kaggle
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content'
'''

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧠 Model Architecture
python
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧪 Model Training

Loss Function: Binary Crossentropy
Optimizer: Adam
Epochs: 10+
Validation Split: 20%

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📊 Results

The model achieves good validation accuracy.
Can successfully classify new images of cats and dogs.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## How to Run
🔗 Run in Google Colab:

🖥️ Run Locally:
1) Clone the repository:
git clone https://github.com/UtkarshSaxena26062004/Image-Classification-of-Cats-and-Dogs-using-CNN-Project
cd Image-Classification-of-Cats-and-Dogs-using-CNN-Project

2) Install dependencies:
pip install -r requirements.txt

3) Download dataset from Kaggle (needs API key as mentioned above).

4) Run the notebook:
Open My_IBM_AI_Project.ipynb using Jupyter Notebook or Colab.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🛠️ Tools & Technologies
Python
TensorFlow / Keras
NumPy, Pandas
Matplotlib
Google Colab
Kaggle API

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📦 Project Structure
📁 Image-Classification-of-Cats-and-Dogs-using-CNN-Project/
├── My_IBM_AI_Project.ipynb         # Main notebook
├── model_catsvsdogs.h5             # Trained model file
├── Data/                           # Dataset directory (not uploaded)
└── README.md                       # This file

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ✍️ Author
Utkarsh Saxena
📍 Moradabad, Uttar Pradesh
🔗 GitHub Profile: https://github.com/UtkarshSaxena26062004

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🎓 Training Note
This project was created by me during my training on Artificial Intelligence (AI) conducted by IBM.
It was part of a PBEL (Project Based Experiential Learning) program focused on hands-on learning through real-world projects.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

🌟 Support
If you find this project useful, please ⭐ star the repository and share it!
------------------------------------------------------------------------------------------------------------------------------------------------------------------

Let me know if you also want:
- A requirements.txt file
- Screenshots of predictions
- A deployment version using Streamlit or Flask
I'm happy to help!
