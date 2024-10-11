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

img = np.array(Image.open('arc2face/assets/examples/freddie.png'))[:,:,::-1]

images, distances, output_json = generate_images(img, app, mapper, pipeline, device)

os.makedirs('recovered', exist_ok=True)
for i, image in enumerate(images):
    image.save(os.path.join('recovered', f'rec{i}.jpg'))
print(distances)
print("Done!")
