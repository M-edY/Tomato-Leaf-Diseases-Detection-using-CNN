# Tomato Leaf Disease Detection 🍅
- An end-to-end computer vision solution designed to identify and classify diseases in tomato leaves. This project leverages a Convolutional Neural Network (CNN) for high-accuracy image classification and a FastAPI backend for real-time inference.

# 🚀 Overview
- Early detection of crop diseases is vital for food security and yield optimization. This repository provides:

- CNN Model: Trained to recognize multiple tomato leaf pathologies.

- FastAPI Integration: A lightweight, high-performance web API to serve the model.

- User-Friendly Interface: Easy-to-use endpoints for uploading images and receiving instant diagnostic results.

# 🛠️ Tech Stack
- #### Deep Learning: TensorFlow / Keras (CNN)

- #### Backend: FastAPI, Uvicorn

- #### Data Analysis: NumPy, Pandas, Matplotlib

# 📊 Dataset & Model
The model is trained on the [Kaggle Tomato leaf disease detection dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf), focusing on the following classes:

Tomato_mosaic_virus, Target_Spot, Bacterial_spot, Tomato_Yellow_Leaf_Curl_Virus, Late_blight, Leaf_Mold, Early_blight, Spider_mites Two-spotted_spider_mite, Tomato___healthy & Septoria_leaf_spot


# 🔧 Installation & Setup
##### Clone the Repository:

```
git clone https://github.com/M-edY/Tomato-Leaf-Diseases-Detection-using-CNN.git
cd Tomato-Leaf-Diseases-Detection-using-CNN
```
##### Running the API
Start the FastAPI server using Uvicorn:
```
uvicorn main:app --reload
```

or just run the ```main.py``` file in the ```api``` folder

Once started, access the interactive documentation (Swagger UI) at:
```http://127.0.0.1:8000/docs```
# 🎥 Preview


<p align="center">
  <video src="./Assets/Demo.mp4" width="100%" controls>
    Your browser does not support the video tag.
  </video>
</p>


# 🤝 Contributing
Contributions are welcome! If you have suggestions for model optimization or UI improvements, feel free to open an issue or submit a pull request.
