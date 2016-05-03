import cv2
import os
from emotion_network import EmotionNetwork
from face_detector import calculate_features

DIR_HAPPY = './data/workarea/happy/'
DIR_NEUTRAL = './data/workarea/neutral/'
DIR_SURPRISE = './data/workarea/surprise/'

USE_CACHE = True

cache = dict()


def normalize(dataset):
    l = list(map(lambda x: (x), dataset))
    return l
    # return preprocessing.scale(l)


def calculate_scaled(path):
    im = cv2.imread(path)
    features = calculate_features(im)
    scaled = normalize(features)
    cache[path] = scaled
    return scaled


def get_scaled(path):
    if path in cache:
        return cache[path]
    return calculate_scaled(path)


def main():
    flag = False

    if flag:
        net = EmotionNetwork(load='3emo.p')
        for i in range(10000):
            for happy, neutral, surprise in zip(
                    os.listdir(DIR_HAPPY), os.listdir(DIR_NEUTRAL),
                    os.listdir(DIR_SURPRISE)):

                scaled = get_scaled(os.path.join(DIR_HAPPY, happy))
                net.train(scaled, [1, 0, 0])

                scaled = get_scaled(os.path.join(DIR_NEUTRAL, neutral))
                net.train(scaled, [0, 1, 0])

                scaled = get_scaled(os.path.join(DIR_SURPRISE, surprise))
                net.train(scaled, [0, 0, 1])

            print('Epoch', i + 1)

        scaled = get_scaled('./data/workarea/farhanhappy.JPG')
        print('happyface: ', net.calculate(scaled))

        scaled = get_scaled('./data/workarea/neutral.png')
        print('neutralface: ', net.calculate(scaled))

        scaled = get_scaled('./data/workarea/surprisenaeem.JPG')
        print('surpriseface: ', net.calculate(scaled))
        net.save('3emo.p')

    else:
        debug = False
        net = EmotionNetwork(load='3emo.p')
        if debug:

            scaled = get_scaled('./data/workarea/farhanhappy.JPG')
            print('happy: ', net.calculate(scaled))

            scaled = get_scaled('./data/workarea/neutral.png')
            print('neutral: ', net.calculate(scaled))

            scaled = get_scaled('./data/workarea/surprisenaeem.JPG')
            print('surprise: ', net.calculate(scaled))

            return

        cam = cv2.VideoCapture(0)
        while True:
            rval, img = cam.read()
            features = calculate_features(img)
            text = 'No face detected'
            if features is not None:
                scaled = normalize(calculate_features(img))
                text = net.predict(scaled)
            cv2.putText(img, text, (250, 100), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (0, 0, 0))

            cv2.imshow('win', img)
            cv2.waitKey(5)


if __name__ == '__main__':
    main()
