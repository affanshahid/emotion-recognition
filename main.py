import numpy as np
import math
import cv2
import os
from emotion_network import EmotionNetwork
from face_detector import calculate_features
from sklearn import preprocessing


def normalize(dataset):
    l = list(map(lambda x: (x), dataset))
    return l
    # return preprocessing.scale(l)


def get_scaled(path):
    im = cv2.imread(path)
    features = calculate_features(im)
    scaled = normalize(features)
    return scaled


def main():
    net = EmotionNetwork(load=True)
    flag = True

    if flag:
        for i in range(1):
            for happy, neutral in zip(
                    os.listdir('./data/workarea/jaffe/happy/'),
                    os.listdir('./data/workarea/jaffe/neutral/')):

                scaled = get_scaled('./data/workarea/jaffe/happy/' + happy)
                net.train(scaled, [1])
                scaled = get_scaled('./data/workarea/jaffe/neutral/' + neutral)
                net.train(scaled, [0])
            print(i)

        scaled = get_scaled('./data/workarea/jaffe/happy/KR.HA2.75.tiff')
        print('happyface: ', net.calculate(scaled))

        scaled = get_scaled('./data/workarea/jaffe/neutral/KR.NE1.71.tiff')
        print('neutralface: ', net.calculate(scaled))
        net.save('2emo.p')
    else:
        scaled = get_scaled('./data/workarea/farhanhappy.JPG')
        print('happy: ', net.calculate(scaled))

        scaled = get_scaled('./data/workarea/farhanneutral.JPG')
        print('neutral: ', net.calculate(scaled))


if __name__ == '__main__':
    main()
