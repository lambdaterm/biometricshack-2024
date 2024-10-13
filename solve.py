from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from arc2face.arc2face import CLIPTextModelWrapper, project_face_embs

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
from utills import EmbeddingMapper, generate_images
import os
import pickle
from tqdm import tqdm
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    )
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(device)

app = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(512, 512))

mapper = EmbeddingMapper(input_dim=512, output_dim=512)
mapper.load_state_dict(torch.load("./models/emb_mapper/emb_mapper.pth"))
mapper.eval()

with open('embedding_v3.pickle', 'rb') as file:
    loaded_list = pickle.load(file)

folder_for_recovered = "recovered_3"
os.makedirs(folder_for_recovered, exist_ok=True)
files_done = os.listdir(folder_for_recovered)
similarities_dict = dict()
for key, value in tqdm(loaded_list.items()):
    if key in files_done:
        print(f"{key} skipped")
        continue
    best_sim = 0.0
    count = 0
    while count < 15:
        images, similarities, output_json = generate_images(app, mapper, pipeline, device,
                                                            tensor=loaded_list[key]["embedding"], num_inferences=25,
                                                            guidance_scale=3.0, num_images=6)
        index_best = np.argmax(similarities)
        if similarities[index_best] > best_sim:
            best_image = images[index_best]
            best_sim = similarities[index_best]
            count = 0
        count += 1

    best_image.save(os.path.join(folder_for_recovered, f'{key}'))
    similarities_dict[key] = best_sim
    print(similarities_dict[key])

with open('similarities_3.json', 'w', encoding='utf-8') as file:
    # Сохраняем словарь с отступами для удобного чтения
    json.dump(similarities_dict, file, ensure_ascii=False, indent=4)
