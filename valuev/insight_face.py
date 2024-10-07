import glob
import pickle
import shutil
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


def custom_key(in_face):
    return in_face.det_score


WORKING_DIR = r'D:\Data\Men_faces_renamed'
# TARGET_DIR = r'D:\Data\Men_faces_renamed'


if __name__ == '__main__':

    # app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # app.prepare(ctx_id=0, det_size=(640, 640))

    # img = cv2.imread('fake_face.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # faces = app.get(img)
    # faces.sort(key=custom_key, reverse=True)
    # vector = faces[0].embedding

    # print(vector)
    # print(len(vector))
    zeros = np.zeros(512)
    # print(empty)
    # print(len(empty))

    face_list = {}
    ext_set = set()

    for cls in os.listdir(WORKING_DIR):
        folder_cls = os.path.join(WORKING_DIR, cls)
        for folder in os.listdir(folder_cls):
            folder_path = os.path.join(folder_cls, folder)
            for file_name in os.listdir(folder_path):

                original_file_name, original_extension = os.path.splitext(file_name)
                file_name_from = os.path.join(folder_path,original_file_name+original_extension)
                file_name_to = os.path.join(folder_path, original_file_name + '.jpg')
                print(file_name_from)
                try:
                    image = cv2.imread(file_name_from)
                    cv2.imwrite(file_name_to, image)
                    if original_extension != '.jpg':
                        os.remove(file_name_from)
                    # image = Image.open(from_file)
                    # image.save(to_file)
                except:
                    os.remove(file_name_from)
                    print('!!!!')


    # for folder in os.listdir(WORKING_DIR):
    #     folder_path = os.path.join(WORKING_DIR, folder)
    #     for file_name in os.listdir(folder_path):
    #
    #         original_file_name, original_extension = os.path.splitext(file_name)

    #         if original_extension == '.Png':
    #             print(original_file_name)
    #         ext_set.add(original_extension)
    #
    # print(ext_set)

            # img_file_name_to = 'real_train_'+folder+'_'+str(cnt)+original_extension
            # cnt += 1
            # img_path_to = os.path.join(TARGET_DIR, img_file_name_to)
            # img_path_from = os.path.join(folder_path, file_name)
            # try:
            #     img = cv2.imread(img_path_from)
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     faces = app.get(img)
            #     faces.sort(key=custom_key, reverse=True)
            #     print(original_file_name, len(faces))
            #     if len(faces) == 0:
            #         shutil.copy(img_path_from, img_path_to)
            #         os.remove(img_path_from)
            # except:
            #     print('!!!!')
            #     os.remove(img_path_from)


        #
        # vector = faces[0].embedding


    #     print(original_file_name.split('_')[0])
    #     key = original_file_name.split('_')[0]
    #     img = cv2.imread(img_path)
    #     try:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         faces = app.get(img)
    #     except:
    #         continue
    #     faces.sort(key=custom_key, reverse=True)
    #     if len(faces) == 0:
    #         continue
    #     vector = faces[0].embedding
    #     face_list[key] = vector
    # # print(face_list['00ab42f9-e86e-4de0-9dd4-a3d7771ecbe8'])
    #
    # # with open('D:\\Data\\Faces\\pickle_small_test.pkl', 'wb') as pickle_file_name:
    # #     pickle.dump(face_list, pickle_file_name)
    # with open('D:\\Data\\Faces\\doc_rgb.pkl', 'wb') as pickle_file_name:
    #     pickle.dump(face_list, pickle_file_name)
    #
    # print(len(face_list))

    # with open('D:\\Data\\Faces\\pickle_small_test.pkl', 'rb') as pickle_file_name:
    #     face_list = pickle.load(pickle_file_name)
    #
    # print(face_list['00ab42f9-e86e-4de0-9dd4-a3d7771ecbe8'])


    # with open('filename', 'wb') as f: pickle.dump(arrayname, f)
    # with open('filename', 'rb') as f: arrayname1 = pickle.load(f)

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

     # for file in glob.iglob(TEST_FOLDER, recursive=True):
     #        print(file)
     #        original_file_name, original_extension = os.path.splitext(file)
     #        # print(original_file_name,original_extension)
     #        # new_file = original_file_name.replace('(', '[').replace(')', ']').replace('.', '^') + original_extension
     #        file_path, file_name = os.path.split(original_file_name)
     #        # print(file_name)
     #        # file_name = file_name.replace('a', '')
     #        new_file = file_path + '\\~' + file_name + original_extension
     #        print(file, new_file)
     #        os.rename(file, new_file)

