import numpy as np
import cv2 as cv2
from PIL import Image, ImageEnhance, ImageOps
import time
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import blynklib
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library

GPIO.setwarnings(False)  # Ignore warning for now
GPIO.setmode(GPIO.BCM)  # Use physical pin numbering
motor1 = 13
motor2 = 12
GPIO.setup(motor1, GPIO.OUT)
GPIO.setup(motor2, GPIO.OUT)
motor1Servo = GPIO.PWM(motor1, 50)
motor1Servo.start(8)
motor2Servo = GPIO.PWM(motor2, 50)
motor2Servo.start(8)
motor1Servo.ChangeDutyCycle(7.5)
motor2Servo.ChangeDutyCycle(7.5)

def servoControl(value): #substitute this with your motor controller
    motor1Servo.ChangeDutyCycle(8.5 + value)
    motor2Servo.ChangeDutyCycle(8.5 - value)

class Agent:
    def __init__(self):
        self.userSteering = 0
        self.aiMode = False
        self.model = Sequential([
            Conv2D(32, (7, 7), input_shape=(240, 320, 3), strides=(2, 2), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid'),
            Conv2D(64, (4, 4), activation='relu', strides=(1, 1), padding='same'),
            MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid'),
            Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid'),
            Flatten(),
            Dense(384, activation='relu'),
            Dense(64, activation="relu", name="layer1"),
            Dense(8, activation="relu", name="layer2"),
            Dense(1, activation="linear", name="layer3"),
        ])
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.05))
        self.model.load_weights("selfdrive.h5")
        self.cap = cv2.VideoCapture(0) #you might need to tweak this based on your camera
        self.cap.set(3, 320) # you might need to tweak this based on your camera
        self.cap.set(4, 240) #you might need to tweak this based on your camera

        self.stateMemory = []
        self.actionMemory = []
        # print(self.model.summary()) # use this for more detailed stats on your model!
        self.xTrain = []
        self.yTrain = []
        self.loss = []
        self.imageBank = []
        self.imageBankLength = 4

    def act(self, state):
        state = np.reshape(state, (1, 240, 320, 3))
        action = self.model.predict(state)[0][0]
        servoControl((action * 2) - 1)
        return (action * 2) - 1

    def learn(self):
        state = self.stateMemory
        action = self.actionMemory
        history = self.model.fit(
            state, action, batch_size=1, epochs=1, verbose=0)
        print("LOSS: ", history.history.get("loss")[0])

    def remember(self, state, action):
        self.stateMemory.append(np.array([state]))
        self.actionMemory.append(np.array([action]))

    def _processImg(self, img):
        state = _imageBankHandler(img) / 255
        # state = generateData(img) # not implemented but you can do so
        # state = _contrast(state) # You can use this. I found it didn't really help
        return state

    def getState(self):
        ret, frame = self.cap.read()
        pic = np.array(frame)
        processedImg = self._processImg(pic)
        return processedImg

    def observeAction(self):
        return (self.userSteering + 1) / 2

    def remember(self, state, action):
        state = self._processImg(state)
        self.memory.append(np.array([state, action]))

    #TODO: optional implementation
    def generateData(self, batch):
        return None
        #image flip
        #change brightness
        
    def _contrast(self, pixvals):
        minval = np.percentile(pixvals, 2)
        maxval = np.percentile(pixvals, 98)
        pixvals = np.clip(pixvals, minval, maxval)
        pixvals = ((pixvals - minval) / (maxval - minval))
        return pixvals

    def _imageBankHandler(self, img):
        return img
        while len(self.imageBank) < (self.imageBankLength):
            self.imageBank.append(np.reshape(img, (240, 320, 3)))

        bank = np.array(self.imageBank)
        toReturn = np.dstack(bank)

        self.imageBank.pop(0)
        self.imageBank.append(np.reshape(img, (240, 320, 3)) * self.ones)

        return toReturn


agent = Agent()  # currently agent is configured with only 2 actions
BLYNK_AUTH = 'MGbFmANkyThXj6e36bE0JnPDWK7V84Xy' #insert your blynk code from your blynk project
blynk = blynklib.Blynk(BLYNK_AUTH)

@blynk.handle_event('write V4')
def write_virtual_pin_handler(pin, value):
    print("value: ", float(value[0]))
    agent.userSteering = float(value[0])
    servoControl(float(value[0]))

@blynk.handle_event('write V2')
def write_virtual_pin_handler(pin, value):
    if value == 1:
        agent.aiMode = False
    else:
        agent.aiMode = True

while True:  
    blynk.run()
    if agent.aiMode == False:
        print("Manual Control")
        start = time.time()
        state = agent.getState()
        action = agent.observeAction()
        if action >= 0:  # if its a valid action
            agent.remember(state, action)
            counter += 1
            if counter % 5 == 0: #change counter % x to change frequency of learning
                start = time.time()
                agent.learn() # you can learn here or before the AI control starts
            if counter % 50 == 0:
               agent.model.save_weights("selfdrive.h5"
        print("framerate: ", 1/(time.time() - start))
    else:
        print("AI Control")
        # agent.learn() # you can learn here or during the training time
        while agent.aiMode == True:
            start = time.time()
            state = agent.getState()
            action = agent.act(state)
            print("AI-action", action)
            print("framerate: ", 1/(time.time() - start))