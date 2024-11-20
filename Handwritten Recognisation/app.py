import pygame
import sys
from pygame.locals import *
import numpy as np
import tensorflow as tf
# from keras.models import load_model
# Specify the path to your saved model file
model_path = "C:/vs code/Data Science/myNotebooks/Projects-Ai-Ml/Handwritten Recognisation/Handwritten_Recognisation.keras"

# Load the model
model = tf.keras.models.load_model(model_path)

# Check the model summary to confirm successful loading
model.summary()

# Specify the path to your saved model file
# model_path = "C://vs code//Data Science//myNotebooks//Projects-Ai-Ml//Handwritten Recognisation//Handwritten_Recognisation.keras"

# # Load the model
# model = load_model(model_path)

# # Check the model summary to confirm successful loading
# model.summary()

import cv2 
Boundaryinc=5
imag_cnt=1
Predict=True

WINDOWSIZEX = 640
WINDOWSIZEY = 480
WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)
IAMGESAVE=False
LABELS={0:"Zero",1:"One",
        2:"Two",3:"Three",
        4:"Four",5:"Five",
        6:"Six",7:"Seven",
        8:"Eight",9:"Nine"}
# Initialize Pygame
pygame.init()

Font=pygame.font.Font(None,18)

# Set up the display
DISPLAYSURF=pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))  # Corrected function call
DISPLAYSURF.fill(BLACK)
pygame.display.set_caption("Digit Board")

iswriting=False
number_Xcord=[]
number_Ycord=[]
# Main loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type== MOUSEMOTION and iswriting:
            Xcord,Ycord=event.pos
            pygame.draw.circle(DISPLAYSURF,WHITE,(Xcord,Ycord),4,0)
            number_Xcord.append(Xcord)
            number_Ycord.append(Ycord)
        
        if event.type==MOUSEBUTTONDOWN:
            iswriting=True
        if event.type==MOUSEBUTTONUP:
            iswriting=False
            if number_Xcord and number_Ycord:
                number_Xcord=sorted(number_Xcord)
                number_Ycord=sorted(number_Ycord)
                # Rect_min_x,Rect_max_x = max(number_Xcord[0]-Boundaryinc,0),min(WINDOWSIZEX,number_Xcord[-1]+Boundaryinc)
                # Rect_min_y,Rect_max_y = max(number_Ycord[0]-Boundaryinc,0),min(WINDOWSIZEY,number_Ycord[-1]+Boundaryinc)
                Rect_min_x = max(number_Xcord[0] - Boundaryinc, 0)
                Rect_max_x = min(number_Xcord[-1] + Boundaryinc, WINDOWSIZEX)
                Rect_min_y = max(number_Ycord[0] - Boundaryinc, 0)
                Rect_max_y = min(number_Ycord[-1] + Boundaryinc, WINDOWSIZEY)
                number_Xcord=[]
                number_Ycord=[]
                
                img_array=np.array(pygame.PixelArray(DISPLAYSURF))[Rect_min_x:Rect_max_x,Rect_min_y:Rect_max_y].T.astype(np.float32)
                if IAMGESAVE:
                    cv2.imwrite("image.png")
                    imag_cnt +=1
                if Predict:
                    image=cv2.resize(img_array,(28,28))
                    image = np.pad(image, ((10, 10), (10, 10)), mode='constant', constant_values=0)
                    image=cv2.resize(image,(28,28))/255
                    
                    lable=str(LABELS[np.argmax(model.predict(image.reshape(1,28,28,1)))])
                    textSurface=Font.render(lable,True,RED,WHITE)
                    textRecObj=textSurface.get_rect()
                    textRecObj.left,textRecObj.bottom=Rect_min_x,Rect_max_y
                    
                    pygame.draw.rect(DISPLAYSURF, RED, pygame.Rect(Rect_min_x, Rect_min_y, Rect_max_x - Rect_min_x, Rect_max_y - Rect_min_y), width=2)
                    
                    DISPLAYSURF.blit(textSurface,textRecObj)
                
                # if event.type==KEYDOWN :
                #     if event.unicode=="n":
                #         DISPLAYSURF.fill(BLACK)
        if event.type == KEYDOWN and event.unicode == "n":
            DISPLAYSURF.fill(BLACK) 
               
                    
    pygame.display.update()