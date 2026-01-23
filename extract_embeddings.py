import cv2, os, numpy as np, pickle

detector = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

embedder = cv2.dnn.readNetFromTorch(
    "models/openface_nn4.small2.v1.t7"
)

embeddings = []
names = []

for person in os.listdir("dataset"):
    person_path = os.path.join("dataset", person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )

        detector.setInput(blob)
        detections = detector.forward()

        if detections.shape[2] > 0:
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(
                cv2.resize(face, (96, 96)), 1.0/255, (96, 96)
            )

            embedder.setInput(face_blob)
            vec = embedder.forward()

            embeddings.append(vec.flatten())
            names.append(person)

data = {"embeddings": embeddings, "names": names}

with open("embeddings.pickle", "wb") as f:
    pickle.dump(data, f)

print("Embeddings extracted successfully")
