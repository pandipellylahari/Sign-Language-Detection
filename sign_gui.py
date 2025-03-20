import tkinter as tk
from tkinter import filedialog, Label, Button
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

# Check if model exists
if not os.path.exists("sign_language_model"):
    print("Error: Model not found! Please train and save the model using model.save('sign_language_model')")
    exit()

# Load trained sign language model
model = load_model("/home/user/Desktop/Sign_Language_Detection/sign_mnist_train")

# List of sign language classes (modify based on your dataset labels)
SIGN_LIST = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

top = tk.Tk()
top.geometry('800x600')
top.title('Sign Language Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def detect_sign(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32)) / 255.0
    image = np.expand_dims(image, axis=[0, -1])
    pred = np.argmax(model.predict(image))
    predicted_sign = SIGN_LIST[pred]
    print(f"Predicted Sign: {predicted_sign}")
    label1.configure(foreground="#011638", text=predicted_sign)

def show_detect_button(file_path):
    detect_b = Button(top, text="Detect Sign", command=lambda: detect_sign(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_detect_button(file_path)
    except Exception as e:
        print("Error loading image:", e)

def live_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (32, 32)) / 255.0
        img = np.expand_dims(img, axis=[0, -1])
        pred = np.argmax(model.predict(img))
        predicted_sign = SIGN_LIST[pred]
        cv2.putText(frame, f'Predicted: {predicted_sign}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Sign Language Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=20)

video_button = Button(top, text="Live Video", command=live_video, padx=10, pady=5)
video_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
video_button.pack(side='bottom', pady=20)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Sign Language Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()

