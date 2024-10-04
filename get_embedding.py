import os

import cv2
from insightface.app import FaceAnalysis

if __name__ == '__main__':
    model_pack_name = 'buffalo_l'
    app = FaceAnalysis(name=model_pack_name)
    app.prepare(ctx_id=0, det_size=(640, 640))

    image = cv2.imread(os.path.join('images', 'Aaron_Eckhart_0001.jpg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(image)
    for face in faces:
        emb = face.normed_embedding
        print(emb)
        print(f'len(emb): {len(emb)}')
