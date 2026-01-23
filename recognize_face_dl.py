import cv2
import numpy as np
import pickle
from collections import deque

detector = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

embedder = cv2.dnn.readNetFromTorch(
    "models/openface_nn4.small2.v1.t7"
)

with open("embeddings.pickle", "rb") as f:
    data = pickle.load(f)

cam = cv2.VideoCapture(0)

def cosine_dist(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

conf_hist = deque(maxlen=5)
frame_count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )

    detector.setInput(blob)
    detections = detector.forward()

    for i in range(detections.shape[2]):
        det_conf = detections[0, 0, i, 2]
        if det_conf > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(
                cv2.resize(face, (96, 96)), 1.0/255, (96, 96)
            )

            embedder.setInput(face_blob)
            vec = embedder.forward().flatten()

            dists = [cosine_dist(vec, e) for e in data["embeddings"]]
            idx = np.argmin(dists)

            dist = dists[idx]
            confidence = (1 - dist) * 100

            conf_hist.append(confidence)
            avg_conf = sum(conf_hist) / len(conf_hist)

            if dist < 0.5:
                name = data["names"][idx]
            else:
                name = "Unknown"

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Distance: {dist:.3f} | Confidence: {avg_conf:.2f}%")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({avg_conf:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Deep Learning Face Recognition", frame)
    # if cv2.waitKey(1) == 27:
    #     break
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if cv2.getWindowProperty("Deep Learning Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break


cam.release()
cv2.destroyAllWindows()
