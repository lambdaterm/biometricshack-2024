import torch.nn as nn
import torch
import cv2
import numpy as np
from arc2face.arc2face import project_face_embs
from scipy import spatial


class EmbeddingMapper(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super(EmbeddingMapper, self).__init__()
        self.lin1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024))
        self.lin2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048))
        self.lin3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024))
        self.lin4 = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        return x


def custom_key(in_face):
    return in_face.det_score


def calculate_cosine_distance(a, b):
    cosine_distance = float(spatial.distance.cosine(a, b))
    return cosine_distance


def calculate_cosine_similarity(a, b):
    cosine_similarity = 1 - calculate_cosine_distance(a, b)
    return cosine_similarity


def generate_images(app, mapper, pipeline, device, img: np.ndarray | None = None, tensor=None, num_images=10, num_inferences=25, guidance_scale=3.0):
    set_of_images = []
    set_of_similarities = []

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img)
        faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)

        template_tensor = faces['embedding']    # buffalo_l embedding
    elif tensor is not None:
        template_tensor = tensor
    else:
        raise AssertionError('No image or tensor provided')

    mapped_emb = mapper(torch.tensor(template_tensor, dtype=torch.float32)[None])  # reconstructed antelopev2 embedding
    id_emb = mapped_emb.to(torch.float16).to(device)
    id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)
    id_emb = project_face_embs(pipeline, id_emb)

    images = pipeline(prompt_embeds=id_emb, num_inference_steps=num_inferences, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images   # generate new faces

    output_json = {
    "initial_photo": {"Id": 1, "Embedding_buffalo_l": np.array(template_tensor).tolist(),
                      "Reconstructed_antelopev2_emb": mapped_emb.detach().cpu().numpy().tolist()[0], "Cosine similarity": 1},

    }

    cnt = 1
    for image in images:
        fcs = app.get(np.array(image))
        fcs.sort(key=custom_key, reverse=True)
        if len(fcs) > 0:
            vector = fcs[0].embedding
            sim = calculate_cosine_similarity(template_tensor, vector)
            output_json["GeneratedImage_"+str(cnt)] = {"Id": cnt, "Embedding_buffalo_l": np.array(vector).tolist(), "Cosine similarity": sim}
            cnt += 1
        else:
            sim = 0

        set_of_images.append(image)
        set_of_similarities.append(sim)

    return set_of_images, set_of_similarities, output_json
