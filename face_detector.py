import dlib
import cv2
import math

FACE_LEFT = 14
FACE_RIGHT = 2

RIGHT_EYE_RIGHT = 36
RIGHT_EYE_LEFT = 39

LEFT_EYE_RIGHT = 42
LEFT_EYE_LEFT = 45

LIP_RIGHT = 48
LIP_LEFT = 54

LIP_TOP = 51
LIP_BOTTOM = 57

RIGHT_EYE_TOP_RIGHT = 37
RIGHT_EYE_TOP_LEFT = 38
RIGHT_EYE_BOTTOM_RIGHT = 41
RIGHT_EYE_BOTTOM_LEFT = 40

LEFT_EYE_TOP_RIGHT = 43
LEFT_EYE_TOP_LEFT = 44
LEFT_EYE_BOTTOM_RIGHT = 47
LEFT_EYE_BOTTOM_LEFT = 46

RIGHT_EYEBROW_TOP = 19
LEFT_EYEBROW_TOP = 24
NOSE_BOTTOM = 33

KEY_REYE_W = 'rew'
KEY_REYE_H = 'reh'

KEY_LEYE_W = 'lew'
KEY_LEYE_H = 'leh'

KEY_MOUTH_H = 'mouh'
KEY_MOUTH_W = 'mouw'

KEY_N_TO_RB = 'ntorb'
KEY_N_TO_LB = 'ntolb'

# 36 right-eye-right 39 right-eye-left
# 42 left-eye-right 45 left-eye-left
# 48 lip-right, 54 lip-left
# 51 lip-top 57 lip-bottom
# 37 right-eye-top-right 38 top-left
# 41 right-eye-bottom-right 40 bottom-left
# 43 left-eye-top-right 44 top-left
# 47 left-eye-bottom-right 46 bottom-left
# 19 right-eyebrow-top 24 left-eyebrow-top
# 33 nose-bottom

predictor_path = './data/predictor/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def convert_rect(rect):
    return [(rect.left(), rect.top()), (rect.right(), rect.bottom())]


def convert_to_points(shape):
    return [(p.x, p.y) for p in shape.parts()]


def calculate_features(im, asList=True):
    fp = get_face_points(im)
    if fp is None:
        return None

    ref = calc_distance(fp[FACE_LEFT], fp[FACE_RIGHT])

    reye_top = calc_midpoint(fp[RIGHT_EYE_TOP_RIGHT], fp[RIGHT_EYE_TOP_LEFT])
    reye_bottom = calc_midpoint(fp[RIGHT_EYE_BOTTOM_RIGHT],
                                fp[RIGHT_EYE_BOTTOM_LEFT])

    reye_width = calc_distance(fp[RIGHT_EYE_RIGHT], fp[RIGHT_EYE_LEFT]) / ref
    reye_height = calc_distance(reye_top, reye_bottom) / ref

    leye_top = calc_midpoint(fp[LEFT_EYE_TOP_RIGHT], fp[LEFT_EYE_TOP_LEFT])
    leye_bottom = calc_midpoint(fp[LEFT_EYE_BOTTOM_RIGHT],
                                fp[LEFT_EYE_BOTTOM_LEFT])

    leye_width = calc_distance(fp[LEFT_EYE_RIGHT], fp[LEFT_EYE_LEFT]) / ref
    leye_height = calc_distance(leye_top, leye_bottom) / ref

    mouth_height = calc_distance(fp[LIP_TOP], fp[LIP_BOTTOM]) / ref
    mouth_width = calc_distance(fp[LIP_RIGHT], fp[LIP_LEFT]) / ref

    nose_to_rb = calc_distance(fp[NOSE_BOTTOM], fp[RIGHT_EYEBROW_TOP]) / ref
    nose_to_lb = calc_distance(fp[NOSE_BOTTOM], fp[LEFT_EYEBROW_TOP]) / ref

    if (asList):
        return [
            reye_height, leye_height, mouth_width, mouth_height, nose_to_rb,
            nose_to_lb
        ]
    else:
        return {
            KEY_REYE_H: reye_height,
            KEY_LEYE_H: leye_height,
            KEY_MOUTH_W: mouth_width,
            KEY_MOUTH_H: mouth_height,
            KEY_N_TO_RB: nose_to_rb,
            KEY_N_TO_LB: nose_to_lb
        }


def calc_distance(tup1, tup2):
    if tup2 is None:
        tup2 = tup1
    return math.hypot(tup1[0] - tup2[0], tup1[1] - tup2[1])


def calc_midpoint(tup1, tup2):
    return ((tup1[0] + tup2[0]) / 2, (tup1[1] + tup2[1]) / 2)


def get_face_points(im):
    faces = detector(im, 1)
    if len(faces) is 0:
        return None
    # get first detected face
    face = faces[0]
    return convert_to_points(predictor(im, face))


def main():
    cam = cv2.VideoCapture(1)
    while True:
        rval, img = cam.read()
        img = cv2.imread('./data/workarea/sample.jpg')
        dets = detector(img, 1)
        if len(dets) > 0:
            face = dets[0]
            shape = predictor(img, face)
            points = convert_to_points(shape)

            for i, p in enumerate(points):
                cv2.circle(img, p, 2, (255, 0, 0), thickness=2)
                cv2.putText(img, str(i), p, cv2.FONT_HERSHEY_PLAIN, 0.7,
                            (0, 0, 255))
                cv2.rectangle(img, *convert_rect(face), 100)
        cv2.imshow('win', img)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()
