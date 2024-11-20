import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2 

# Load model
model_path = "C://vs code//Data Science//myNotebooks//Projects-Ai-Ml//Handwritten Recognisation//Handwritten_Recognisation.keras"
model = load_model(model_path)

# Constants
Boundaryinc = 5
imag_cnt = 1
Predict = True
IAMGESAVE = False

WINDOWSIZEX = 640
WINDOWSIZEY = 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three",
    4: "Four", 5: "Five", 6: "Six", 7: "Seven",
    8: "Eight", 9: "Nine"
}

# Initialize Pygame
pygame.init()
Font = pygame.font.Font(None, 36)  # Default font

# Set up the display
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("Digit Board")

# Variables
iswriting = False
number_Xcord = []
number_Ycord = []

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            Xcord, Ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, BLACK, (Xcord, Ycord), 4, 0)
            number_Xcord.append(Xcord)
            number_Ycord.append(Ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_Xcord = sorted(number_Xcord)
            number_Ycord = sorted(number_Ycord)
            
            if number_Xcord and number_Ycord:  # Ensure lists are not empty
                Rect_min_x = max(number_Xcord[0] - Boundaryinc, 0)
                Rect_max_x = min(WINDOWSIZEX, number_Xcord[-1] + Boundaryinc)
                Rect_min_y = max(number_Ycord[0] - Boundaryinc, 0)
                Rect_max_y = min(WINDOWSIZEY, number_Ycord[-1] + Boundaryinc)

                # Extract and process the drawn image
                img_array = np.array(pygame.PixelArray(DISPLAYSURF))[Rect_min_x:Rect_max_x, Rect_min_y:Rect_max_y].T.astype(np.float32)

                if IAMGESAVE:
                    cv2.imwrite(f"image_{imag_cnt}.png", img_array)
                    imag_cnt += 1

                if Predict:
                    image = cv2.resize(img_array, (28, 28))
                    image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255.0
                    
                    label = str(LABELS[np.argmax(model.predict(image.reshape(1, 28, 28, 1)))])
                    textSurface = Font.render(label, True, RED, WHITE)
                    textRecObj = textSurface.get_rect()
                    textRecObj.left, textRecObj.top = Rect_min_x, Rect_max_y

                    DISPLAYSURF.blit(textSurface, textRecObj)

            # Reset the lists
            number_Xcord = []
            number_Ycord = []

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(WHITE)

    pygame.display.update()