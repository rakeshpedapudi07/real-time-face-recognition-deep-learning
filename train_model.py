import cv2, os, numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = []
labels = []
label_map = {}
label_id = 0

for person in os.listdir("dataset"):
    label_map[label_id] = person
    person_path = os.path.join("dataset", person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label_id)

    label_id += 1

recognizer.train(faces, np.array(labels))
os.makedirs("trainer", exist_ok=True)
recognizer.save("trainer/trainer.yml")

print("Model trained successfully")
