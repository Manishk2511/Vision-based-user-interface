import tkinter as tk
import cv2
import threading

import pandas as pd
import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


datagen = [
ImageDataGenerator(rotation_range = 5, 
                    width_shift_range = 0.15,
                    height_shift_range = 0.15)
]


import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('saved_model')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

stop_processing = threading.Event()


def CustomParser(data):    
    j = np.asarray(data, dtype="int32").flatten()
    return j

def change_color_green(box):
    box.config(bg='green')

def change_border_color_blue(box):
    box.config(bd=5, relief="solid", highlightcolor="blue")

def add_letter(letter):
    letter_label.config(text=letter)

def create_box(root, row, column, letter):
    box = tk.Label(root, text=letter, font=("Arial", 24), width=10, height=5, bg='white')
    box.grid(row=row, column=column)
    box.config()
    box.bind("<Button-1>", lambda event: change_color_green(box))
    box.bind("<Button-3>", lambda event: change_border_color_blue(box))
    return box

def create_rectangle(root):
    rectangle = tk.Label(root, width=20, height=5, bg='white', relief="solid")
    rectangle.grid(row=10, column=0, columnspan=3, pady=10)
    return rectangle


count=0
quadrant=None
user_data={'number':[],'image':[],'quadrant':[]}


def start_calibration():
    boxes[letters[quadrant]].config(bg="blue")
    calibration()
    


def calibration():
    # tk.Message("Please look at the green box for some time")
    # quadrant=0
    global count
    global quadrant
    global user_data
    camera= cv2.VideoCapture(0)
    if(quadrant==None):
        print("Calibration starting......")
        quadrant=0
    # for letter in letters:
    #     count=0
    if(quadrant==9):
        print("All boxes data collected!!! ")
        print(user_data['quadrant'])
        quadrant=0
        return
    
    for j in range(80):
        return_value, image=camera.read()

        img = image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi)

            for (ex, ey, ew, eh) in eyes:
                if ex < w/2:  # Left eye
                    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                    eye_roi = cv2.resize(eye_roi, (100, 100), interpolation=cv2.INTER_LINEAR)
                    image_shape = np.expand_dims(eye_roi, axis=-1).shape
                    data = CustomParser(eye_roi)
                    print("data shape",data.shape)
                    data = np.asarray([np.asarray(data.reshape(image_shape))])
                    print("data shape next",data.shape)
                    data = data / 255
                    print("data shape next next",data.shape)

                    data=np.squeeze(data,axis=0)
                    # print(data.shape) 

                    user_data['number'].append(count)
                    user_data['image'].append(data)
                    user_data['quadrant'].append(quadrant)
                    # print(user_data['quadrant'].shape)
                    count+=1
                    
                    


                else:  # Right eye
                    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                    eye_roi = cv2.resize(eye_roi, (100, 100), interpolation=cv2.INTER_LINEAR)
                    image_shape = np.expand_dims(eye_roi, axis=-1).shape
                    data = CustomParser(eye_roi)
                    print("data shape",data.shape)
                    data = np.asarray([np.asarray(data.reshape(image_shape))])
                    print("data shape next",data.shape)
                    data = data / 255
                    print("data shape next next",data.shape)
                    # data=np.squeeze(data)
                    data=np.squeeze(data,axis=0)

                    # print(data.shape)

                    user_data['number'].append(count)
                    user_data['image'].append(data)
                    user_data['quadrant'].append(quadrant)
                    # print(user_data['quadrant'].shape)
                    count+=1
    box.config(bg='white')
    quadrant+=1
    


def train_calibrate():
    data = pd.DataFrame(user_data)
    data['image'] = data['image'].tolist()
    # x = np.array(data['image'])
    x = np.stack(data['image'])


    print(data['image'][0].shape)
    print(x.shape)
    print(x.shape[0])
    print('printing \n')
    print(len(data['image']))
    for image in data['image']:
        print(image.shape)

    print(x[0].shape)

    # Verify the length of x matches the number of elements
    if len(data['image']) != x.shape[0]:
        print("Mismatch between data['image'] length and x length.")
        

    print('printing here')
    print(x.shape[0])
    print(x.shape)
    y = data.quadrant.to_numpy()
    y=np.asarray([np.asarray(cls.reshape(1)) for cls in y])

    y = to_categorical(y, 9)

    print(y.shape)
    print(y[0].shape)

    print(type(x))
    print(type(y))

    datagen_train = datagen[0].flow(x, y, batch_size=10)

    model.fit(
        datagen_train,
        epochs=50,
        batch_size=64,
        steps_per_epoch=int(np.ceil(len(x) / float(32))),
    )


keystrokes=""
selectedBox=None

def process_image():
    while not stop_processing.is_set():
        global selectedBox

        
        def change_color_white():
            selectedBox.config(bg='white')

        if selectedBox!=None:
            root.after(1000, change_color_white)

        if selectedBox!=None:
            selectedBox.config(bg='white')
        camera= cv2.VideoCapture(0)
        return_value, image=camera.read()

        img = image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        selected = None

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi)

            for (ex, ey, ew, eh) in eyes:
                if ex < w/2:  # Left eye
                    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                    eye_roi = cv2.resize(eye_roi, (100, 100), interpolation=cv2.INTER_LINEAR)
                    image_shape = np.expand_dims(eye_roi, axis=-1).shape
                    data = CustomParser(eye_roi)
                    data = np.asarray([np.asarray(data.reshape(image_shape))])
                    data = data / 255

                    predictions = model.predict(data)
                    max_value = np.amax(predictions[0])
                    result = np.where(predictions[0] == np.amax(predictions[0]))[0]
                    selected = result

                else:  # Right eye
                    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                    eye_roi = cv2.resize(eye_roi, (100, 100), interpolation=cv2.INTER_LINEAR)
                    image_shape = np.expand_dims(eye_roi, axis=-1).shape
                    data = CustomParser(eye_roi)
                    data = np.asarray([np.asarray(data.reshape(image_shape))])
                    data = data / 255

                    predictions = model.predict(data)
                    max_value = np.amax(predictions[0])
                    result = np.where(predictions[0] == np.amax(predictions[0]))[0]
                    selected = result
                    print(result)


        if selected is not None:
            # if selected[0]==0:
            #     selected[0]=2
            # if selected[0]==2:
            #     selected[0]=0
            # if selected[0]==3:
            #     selected[0]=5
            # if selected[0]==5:
            #     selected[0]=3
            # if selected[0]==6:
            #     selected[0]=8
            # if selected[0]==8:
            #     selected[0]=6
            selectedBox=boxes[letters[selected[0]]]

            def change_color():
                change_color_green(selectedBox)

            # Schedule the color change after a delay (e.g., 100 milliseconds)
            root.after(100, change_color)
            
            

            global keystrokes
            keystrokes+=letters[selected[0]]
            add_letter(keystrokes)

        
        # def start_delay():
        #     root.after(10000) 
        # time.sleep(1)

        # start_delay()



def start_parallel_processing():
    stop_processing.clear()
    processing_thread = threading.Thread(target=process_image)
    processing_thread.start()


def create_start_button():
    start_frame = tk.Frame(root)
    start_frame.grid(row=11, column=0, columnspan=3, pady=10)

    start_button = tk.Button(start_frame, text="Start Eye Typing", command=start_parallel_processing)
    
    start_button.pack()


def stop_parallel_processing():
    stop_processing.set()


def create_stop_button():
    stop_frame = tk.Frame(root)
    stop_frame.grid(row=11, column=1, columnspan=5, pady=10)

    stop_button = tk.Button(stop_frame, text="Stop Eye Typing", command=stop_parallel_processing)
    stop_button.pack()


def create_calibrate_button():
    calibrate_frame=tk.Frame(root)
    calibrate_frame.grid(row=12,column=0,columnspan=3,pady=10)
    calibrate_button=tk.Button(calibrate_frame, text="Calibrate", command=calibration)
    calibrate_button.pack()


def create_train_calibration_button():
    calibrate_train_frame=tk.Frame(root)
    calibrate_train_frame.grid(row=12,column=1,columnspan=3,pady=10)
    calibrate_button=tk.Button(calibrate_train_frame, text="Train on calibration data", command=train_calibrate)
    calibrate_button.pack()

root = tk.Tk()
root.title("EyeType App")



# Create the boxes
boxes = {}
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
for i, letter in enumerate(letters):
    row = i // 3
    column = i % 3
    box = create_box(root, row, column, letter)
    boxes[letter] = box

# Create the rectangle
letter_label = create_rectangle(root)

# Example usage:
# add_letter('X')  # Add 'X' to the rectangle

create_start_button()
create_calibrate_button()
create_train_calibration_button()
create_stop_button()





root.mainloop()
