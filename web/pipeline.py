from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from emb_mapper import EmbeddingMapper
from typing import List
import torch
import cv2
import numpy
import json
import onnxruntime as ort
from arc2face import CLIPTextModelWrapper, project_face_embs
from insightface.app import FaceAnalysis
from PIL import Image
from scipy import spatial
import numpy as np


base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=torch.float16
)

unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=torch.float16
)

pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    # , device_map='auto'
    ).to(torch.device('cuda'))

mapper = EmbeddingMapper(input_dim=512, output_dim=512)
mapper.load_state_dict(torch.load("./models/emb_mapper/emb_mapper.pth"))
mapper.eval()


def calculate_cosine_distance(a, b):
    cosine_distance = float(spatial.distance.cosine(a, b))
    return cosine_distance


def calculate_cosine_similarity(a, b):
    cosine_similarity = 1 - calculate_cosine_distance(a, b)
    return cosine_similarity


def custom_key(in_face):
    return in_face.det_score


# pip install torch --index-url https://download.pytorch.org/whl/cu121
# P = np.linalg.lstsq(X_homo, Y, rcond=None)[0].T  # Affine matrix. 3 x 4 transform.py
if __name__ == '__main__':

    print(ort.get_available_providers())

    app = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(r'../images/face.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img)

    # img = np.array(Image.open('assets/examples/joacquin.png'))[:,:,::-1]
    # faces = app.get(img)

    faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)

    template_tensor = faces['embedding']

    id_emb = torch.tensor(faces['embedding'], dtype=torch.float32)[None]

    id_emb = mapper(id_emb)
    id_emb = id_emb.to(torch.float16).to(torch.device('cuda'))
    id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding

    id_emb = project_face_embs(pipeline, id_emb)    # pass through the encoder

    num_images = 4
    images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images

    # print(len(images))

    for image in images:

        cv_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        # cv2.imshow("Generated", cv_image)

        fcs = app.get(cv_image)
        fcs.sort(key=custom_key, reverse=True)
        if len(fcs) > 0:
            vector = fcs[0].embedding
            output_json = {
                "Image_photo": {"Id": 1, "Normed_embedding": np.array(vector).tolist()}
            }
