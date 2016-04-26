import numpy as np
import cv2
import math
import os
from emotion_network import EmotionNetwork

width, height = 28, 10
pixels = width * height


def vectorize(filename):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im_resized = cv2.resize(im, (width, height))
    return im_resized.flatten().tolist()


def main():
    
    cascade = cv2.CascadeClassifier(cv2.FACE)
    im = cv2.imread('./data/neutral/f4003-m.jpg')
    mouth = cascade.detectMultiScale(im)
    print(mouth)


if __name__ == '__main__':
    main()
