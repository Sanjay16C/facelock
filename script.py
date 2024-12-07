import cv2
import os

def capture_dataset(name):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('/Users/sanjayc/DEV/Projects/Faceid/haarcascade_frontalface_default.xml')
    count = 0
    save_path = f"dataset/known_faces/{name}"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"Capturing images for {name}. Press 'q' to quit.")
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y+h, x:x+w]
            file_name = f"{save_path}/{name}_{count}.jpg"
            cv2.imwrite(file_name, face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('Capturing Dataset', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 1200:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Dataset created for {name} with {count} images.")

if __name__ == "__main__":
    user_name = input("Enter the name of the authorized user: ")
    capture_dataset(user_name)