import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load mask detector model (.h5 format)
model = load_model("mask_detector.model.h5")

# Create app window
window = tk.Tk()
window.title("Real-Time Mask Detector")
window.geometry("800x600")

# Video frame
video_label = Label(window)
video_label.pack()

cap = cv2.VideoCapture(0)

def detect_and_display():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        (mask, withoutMask) = model.predict(face_array)[0]

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Convert image to Tkinter format
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Repeat after delay
    video_label.after(10, detect_and_display)

# Start detection loop
detect_and_display()
window.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
