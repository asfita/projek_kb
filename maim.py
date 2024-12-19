import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# Fungsi untuk mendapatkan gambar wajah dan label
def get_images_and_labels(main_path=r'D:\UAS MK KB\dataset_webcam\dataset_webcam'):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = []
    labels = []
    label_names = {}
    current_label = 0

    for folder_name in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder_name)

        if os.path.isdir(folder_path):
            label_names[current_label] = folder_name
            print(f"Processing folder: {folder_name} with label {current_label}")

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Skipping invalid image: {image_path}")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

                for (x, y, w, h) in faces_detected:
                    face_resized = cv2.resize(gray[y:y+h, x:x+w], (150, 150))
                    faces.append(face_resized)
                    labels.append(current_label)

            current_label += 1

    return faces, labels, label_names

# Ambil gambar wajah dan label
faces, labels, label_names = get_images_and_labels()

# Jika dataset tersedia, lanjutkan
if len(faces) > 0:
    # Preprocessing data wajah menggunakan HOG
    faces_hog = [hog(face, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys') for face in faces]

    # Split dataset menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(faces_hog, labels, test_size=0.2, random_state=42)

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Train SVM classifier dengan kernel RBF
    clf = SVC(kernel='rbf', probability=True, gamma='scale', C=10)
    clf.fit(X_train, y_train_encoded)

    # Evaluasi model
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred):.2f}")

    # Simpan model pelatihan
    np.save('svm_model.npy', clf)
    np.save('label_encoder.npy', le)
else:
    print("Dataset kosong atau tidak valid. Pastikan dataset berisi gambar wajah.")
    exit()

# Load model yang telah disimpan
clf = np.load('svm_model.npy', allow_pickle=True).item()
le = np.load('label_encoder.npy', allow_pickle=True).item()

# Inisialisasi deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mulai kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat mengakses webcam. Pastikan webcam terhubung.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    for (x, y, w, h) in faces_detected:
        face_resized = cv2.resize(gray[y:y+h, x:x+w], (150, 150))
        face_hog = hog(face_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)

        label_encoded = clf.predict(face_hog)
        proba = clf.predict_proba(face_hog)
        confidence = np.max(proba)

        label = le.inverse_transform(label_encoded)[0]
        name = label_names.get(label, "Unknown")

        cv2.putText(frame, f"{name} ({int(confidence * 100)}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
