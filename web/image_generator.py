import cv2
import torch
import numpy as np
from typing import List
from scipy import spatial
from arc2face import CLIPTextModelWrapper, project_face_embs


def custom_key(in_face):
    return in_face.det_score


def calculate_cosine_distance(a, b):
    cosine_distance = float(spatial.distance.cosine(a, b))
    return cosine_distance


def calculate_cosine_similarity(a, b):
    cosine_similarity = 1 - calculate_cosine_distance(a, b)
    return cosine_similarity


def gluing_func(imgs: List[np.ndarray], size: int, cos_distances: List[float]) -> np.ndarray:
    if len(imgs) != 4:
        print('Изображений не 4')
    if len(cos_distances) != 4:
        print('Дистанций не 4')
    row1 = cv2.hconcat([imgs[0], imgs[1]])
    row2 = cv2.hconcat([imgs[2], imgs[3]])
    square = cv2.vconcat([row1, row2])
    final_img = cv2.resize(square, (size, size))
    k = size/560
    if size < 512: width = 1
    elif size < 900: width = 2
    else: width = 3
    cv2.putText(final_img, f'{cos_distances[0]:.2f}', (2, round(k*30)), cv2.FONT_HERSHEY_SIMPLEX,
            k, (255, 36, 0), width, cv2.LINE_AA)
    cv2.putText(final_img, f'{cos_distances[1]:.2f}', (size//2+2, round(k*30)), cv2.FONT_HERSHEY_SIMPLEX,
            k, (255, 36, 0), width, cv2.LINE_AA)
    cv2.putText(final_img, f'{cos_distances[2]:.2f}', (2, round(k*30)+size//2), cv2.FONT_HERSHEY_SIMPLEX,
            k, (255, 36, 0), width, cv2.LINE_AA)
    cv2.putText(final_img, f'{cos_distances[3]:.2f}', (size//2+2, round(k*30)+size//2), cv2.FONT_HERSHEY_SIMPLEX,
            k, (255, 36, 0), width, cv2.LINE_AA)
    return final_img


def generate_images(img: np.ndarray, app, pipeline):
    set_of_images = []
    set_of_distances = []

    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img)
        faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)

        template_tensor = faces['embedding']

        id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].cuda()
        id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)
        id_emb = project_face_embs(pipeline, id_emb)

        num_images = 4
        images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images

        output_json = {
            "initial_photo": {"Id": 1, "Normed_embedding": np.array(template_tensor).tolist(), "Cosine similarity": 1},
        }

        cnt = 1
        for image in images:
            fcs = app.get(np.array(image))
            fcs.sort(key=custom_key, reverse=True)
            if len(fcs) > 0:
                vector = fcs[0].embedding
                sim = calculate_cosine_similarity(template_tensor, vector)
                output_json["GeneratedImage_"+str(cnt)] = {"Id": cnt, "Normed_embedding": np.array(vector).tolist(), "Cosine similarity": sim}
                cnt += 1
            else:
                sim = 0

            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            set_of_images.append(cv_image)
            set_of_distances.append(sim)

        glued_image = gluing_func(set_of_images, 900, set_of_distances)

    except:
        glued_image = img
        output_json = {
            "Error": "Couldn't parse an image"
        }

    return glued_image, output_json
