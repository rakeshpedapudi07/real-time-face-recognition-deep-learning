
# Real-time-face-recognition-Deep-learning

Deep learning based real-time face recognition using OpenCV.

---

## 🔍 Project Overview

This project implements a **real-time face recognition system** using deep learning and OpenCV.  
It detects faces from a live webcam feed, extracts facial embeddings using a pre-trained neural network, and recognizes known individuals with a confidence score.

The system is designed to work reliably even under **moderate and low-light conditions**.

---

## 🧠 Technologies Used

- Python
- OpenCV (cv2)
- Deep Learning (OpenCV DNN module)
- OpenFace (Face Embedding Model)
- NumPy
- Pickle

---

## 🧩 Models Used

| Task | Model |
|----|----|
| Face Detection | SSD + ResNet (Caffe) |
| Face Embedding | OpenFace (`openface_nn4.small2.v1.t7`) |

---

## 📁 Project Structure
```
face_project/
│
├── models/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── openface_nn4.small2.v1.t7
│
├── dataset_creator.py
├── extract_embeddings.py
├── train_model.py
├── recognize_face.py
├── recognize_face_dl.py
│
├── camera_test.py
├── face_detect_test.py
│
├── embeddings.pickle
├── haarcascade_frontalface_default.xml
│
├── README.md
└── .gitignore
```

---

## 🚀System Workflow

1. **Dataset Creation**
   - Capture multiple face images per person using webcam
   - Images are stored class-wise (one folder per person)

2. **Embedding Extraction**
   - Each face is passed through the OpenFace model
   - 128-dimensional embeddings are generated

3. **Model Training**
   - Embeddings are stored in `embeddings.pickle`

4. **Real-Time Recognition**
   - Face detection using SSD + ResNet
   - Face recognition using cosine similarity
   - Name and confidence score displayed in real time

---

## ▶️ How to Run

1️⃣ Create Face Dataset
``
python dataset_creator.py ``
2️⃣ Extract Face Embeddings
``
python extract_embeddings.py``

3️⃣ Train the Model
``
python train_model.py``

4️⃣ Run Real-Time Face Recognition
``
python recognize_face_dl.py``


Press `ESC` to exit the application.

 ## 📊 Confidence Calculation

Cosine distance is used to compare face embeddings:

``confidence = (1 - distance) × 100``


Lower distance indicates higher similarity and confidence.

 ## 🔐 Privacy & Ethics

Face images are not uploaded to GitHub

Only trained models and code are shared

Dataset remains local to the system

 ## ⚠️ Limitations

Accuracy depends on dataset quality

Extreme lighting or occlusion may reduce recognition accuracy

Not intended for production-level surveillance

 ## 🔮 Future Enhancements

Face alignment for improved accuracy

Liveness detection

GPU acceleration

Web or mobile deployment

Multi-user large-scale dataset support

 ## 👨‍💻 Author

Rakesh Pedapudi
B.Tech (Artificial Intelligence)
Focused on Computer Vision and Deep Learning

 ## 📄 License

This project is licensed under the MIT License.


---
