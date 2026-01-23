import cv2, os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

label_map = {}
label_id = 0
for person in os.listdir("dataset"):
    label_map[label_id] = person
    label_id += 1

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        print(confidence)

        if confidence < 100:
            name = label_map[id_]
        else:
            name = "Not in Database Unknown"

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition",img)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
