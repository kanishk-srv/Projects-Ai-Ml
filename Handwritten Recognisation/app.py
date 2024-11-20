import pygame
import sys
from pygame.locals import *
import numpy as np
# from keras.models import load_model
# Specify the path to your saved model file
model_path = "C://vs code//Data Science//myNotebooks//Projects-Ai-Ml//Handwritten Recognisation//Handwritten_Recognisation.keras"

# Load the model
model = load_model(model_path)

# Check the model summary to confirm successful loading
model.summary()
from keras.models import load_model

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
LABLES={0:"Zero",1:"One",
        2:"Two",3:"Three",
        4:"Four",5:"Five",
        6:"Six",7:"Seven",
        8:"Eight",9:"Nine"}
# Initialize Pygame
pygame.init()
# Font=pygame.font.Font("DMSansBold.tff,18")

# Set up the display
DISPLAYSURF=pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))  # Corrected function call
DISPLAYSURF.fill(WHITE)
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
            number_Xcord=sorted(number_Xcord)
            number_Ycord=sorted(number_Xcord)
            Rect_min_x,Rect_max_x = max(number_Xcord[0]-Boundaryinc,0),min(WINDOWSIZEX,number_Xcord[-1]+Boundaryinc)
            Rect_min_y,Rect_max_y = max(number_Ycord[0]-Boundaryinc),min(Boundaryinc+number_Xcord[-1]+WINDOWSIZEX)
            
            number_Xcord=[]
            number_Ycord=[]
            
            img_array=np.array(pygame.pixelarray(DISPLAYSURF))[Rect_min_x:Rect_max_x,Rect_min_y:Rect_max_y].T.astype(np.float32)
            if IAMGESAVE:
                cv2.imwrite("image.png")
                imag_cnt +=1
            if Predict:
                image=cv2.resize(img_array,(28,28))
                image=np.pad(image(10,10),'constant',constant_values=0)
                image=cv2.resize(imag,(28,28))/255
                
                lable=str(LABLES[np.argmax(model.predict(image.reshape(1,28,28,1)))])
                textSurface=Font.render(lable,True,RED,WHITE)
                textRecObj=testing.get_rect()
                textRecObj.left,textRecObj.bottom=Rect_min_x,Rect_max_y
                
                DISPLAYSURF.blit(textSurface,textRecObj)
            
            if event.type==KEYDOWN:
                if event.unicode=="n":
                    DISPLAYSURF.fill(BLACK)    
                    
    pygame.display.update()