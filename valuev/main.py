import cv2
import numpy as np
import torch
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import onnxruntime as ort

np.int = np.int32
np.float = np.float64
np.bool = np.bool_


def custom_key(in_face):
    return in_face.det_score


if __name__ == '__main__':

    print(ort.get_available_providers())

    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # img = ins_get_image('t1')
    # faces = app.get(img)
    # rimg = app.draw_on(img, faces)
    # cv2.imshow("Detected", rimg)
    # cv2.waitKey(0)




    #
    # app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # app.prepare(ctx_id=0, det_size=(640, 640))
    #
    img = cv2.imread('../images/face.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = app.get(img)
    faces.sort(key=custom_key, reverse=True)

    vector = faces[0].embedding
    print(vector)
    print(vector[:5])


    # img = cv2.imread('D:\\Data\\Faces\\Test\\000bf0d6-79d3-4816-ba21-eea27a9688d6_Normal_Front.jpg')
    # faces = app.get(img)
    # # print(faces[0].embedding[0])
    # faces.sort(key=custom_key, reverse=True)
    # print(faces[0].embedding[0])

    # face = faces[0]
    # print(np.shape(face.embedding))
    # print(face.embedding)
    # rimg = app.draw_on(img, faces)
    # cv2.imwrite('D:\\Data\\Faces\\Test\\D3_face.jpg', rimg)


