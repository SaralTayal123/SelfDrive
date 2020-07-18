import numpy as np #Used for processing image data
import cv2 as cv2 #Used for capturing images with the webcam
from PIL import Image, ImageEnhance, ImageOps #Used for processing image data
import time #Used for an FPS counter
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D #Import various layers for the Neural Net
from tensorflow.keras.models import Sequential #Import libraries for the Neural Net
from tensorflow.keras.optimizers import Adam  #Import optimizers for the Neural Net
import tensorflow as tf #import Tensorflow
from tensorflow.keras.models import model_from_json #Import tensorflow file saving
import blynklib #Import Blynk library
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # Use physical pin numbering
motor1 = 13 #GPIO Pin for motor1
motor2 = 12 #GPIO Pin for motor2
GPIO.setup(motor1, GPIO.OUT)
GPIO.setup(motor2, GPIO.OUT)
motor1Servo = GPIO.PWM(motor1, 50) #Set Motors to PWM. Change this depeneding on how your motor controller works
motor1Servo.start(8)
motor2Servo = GPIO.PWM(motor2, 50) #Set Motors to PWM. Change this depeneding on how your motor controller works
motor2Servo.start(8)
motor1Servo.ChangeDutyCycle(7.5) #Had to change duty cycle twice due to a strange bug with my RPI
motor2Servo.ChangeDutyCycle(7.5) #Had to change duty cycle twice due to a strange bug with my RPI

def servoControl(value): #Change this function to be apprpriate for your robot and motor controllers
    motor1Servo.ChangeDutyCycle(7.5 + value) # I used this calculation due to tank steering
    motor2Servo.ChangeDutyCycle(7.5 - value)

class Agent:
    def __init__(self):
        self.userSteering = 0
        self.aiMode = False
        self.model = Sequential([ #This is the actual Neural net
            Conv2D(32, (7,7), input_shape=(240, 320, 3),
                   strides=(2, 2), activation='relu', padding = 'same'),
            MaxPooling2D(pool_size=(5,5), strides=(2, 2), padding= 'valid'),
            Conv2D(64, (4, 4), activation='relu', strides=(1, 1), padding = 'same'),
            MaxPooling2D(pool_size=(4,4), strides=(2,2), padding= 'valid'),
            Conv2D(128, (4,4), strides=(1, 1), activation='relu', padding = 'same'),
            MaxPooling2D(pool_size=(5, 5), strides=(3, 3),padding = 'valid'),
            Flatten(),
            Dense(384, activation='relu'),
            Dense(64, activation="relu", name="layer1"),
            Dense(8, activation="relu", name="layer2"),
            Dense(1, activation="linear", name="layer3"),
        ])
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.05))
        #self.model.load_weights("selfdrive.h5") # use this to import your pretrained weights
        # print(self.model.summary()) # use this to get a summary of the CNN
        self.cap = cv2.VideoCapture(0) # This controls which camera device is used. You might need to tweak this based on your camera
        self.cap.set(3, 320) # This controls the camera's resolution. You might need to tweak this based on your camera. 
        self.cap.set(4, 240) # This controls the camera's resolution. You might need to tweak this based on your camera

    def act(self, state): #This method is for the AI behaving in autonomous mode
        state = np.reshape(state, (1, 240, 320, 3))
        action = self.model.predict(state)[0][0]
        action = (action * 2) - 1
        servoControl(action)
        return action

    def learn(self, state, action): #This method is where the AI's Neural net improves/learns
        state = np.reshape(state, (1, 240,320,3))
        history = self.model.fit(state, [action], batch_size=1, epochs=1, verbose=0)
        print("LOSS: ", history.history.get("loss")[0])

    def getState(self):
        ret, frame = self.cap.read() # This gets the actual Webcam Image
        pic = np.array(frame) #Convert it to a NP array
        processedImg = np.reshape(pic, (240, 320, 3)) / 255 #reshape it
        return processedImg 

    def observeAction(self):
        return (self.userSteering + 1) / 2

agent = Agent() 
BLYNK_AUTH = 'insert your code here' #insert your blynk code from your blynk project
blynk = blynklib.Blynk(BLYNK_AUTH)

@blynk.handle_event('write V4') # We used pin v4 on the blynk app for steering control. Hence 'write V4'
def write_virtual_pin_handler(pin, value):
    print("value: ",float(value[0])) 
    agent.userSteering = float(value[0]) #updates the AI's memory of steering angle
    servoControl(float(value[0])) #changes the motors to appropriately turn based on the steering input

@blynk.handle_event('write V2') # We used pin v2 on the blynk app for autonomous/learning control. Hence 'write V2'
def write_virtual_pin_handler(pin, value):
    agent.aiMode = False if value == 1 else True #change the AI's mode based on the reading 

counter = 0
while True: #This is the learning loop
    blynk.run()
    if agent.aiMode == False: #This is the AI's Learning mode
        start = time.time()
        state = agent.getState()
        action = agent.observeAction()
        counter += 1
        if counter % 1 == 0: # you can change this so the AI doesn't learn every iteration
            start = time.time()
            agent.learn(state, action)
            agent.memory = []
        if counter % 50 == 0: #change this to how often you want your AI to save its weights
           agent.model.save_weights("selfdrive.h5")
        print("framerate: ", 1/(time.time() - start))
    else: 
        while agent.aiMode == True: #This is the autonomous loop
            start = time.time() 
            state = agent.getState()
            action = agent.act(state)
            print("action", action)
            print("framerate: ", 1/(time.time() - start))
