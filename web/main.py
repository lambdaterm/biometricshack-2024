import torch
import onnxruntime as ort
from arc2face import CLIPTextModelWrapper, project_face_embs
from insightface.app import FaceAnalysis
from emb_mapper import EmbeddingMapper

import time
from flask import Flask, request, jsonify, make_response, render_template
from flask import request
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import numpy as np
import logging
import json
import base64
from image_generator import generate_images
import json
from wsgiserver import WSGIServer
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

flask_app = Flask(__name__)

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

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


app = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(224, 224))

mapper = EmbeddingMapper(input_dim=512, output_dim=512)
mapper.load_state_dict(torch.load("./models/emb_mapper/emb_mapper.pth"))
mapper.eval()



def custom_key(in_face):
    return in_face.det_score


def add_cors_to_response(in_request, out_response):
    out_response.headers.add("Access-Control-Allow-Credentials", "true")
    out_response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    out_response.headers.add("Access-Control-Allow-Headers", "Origin, Content-Type, Accept")

    if 'Origin' in request.headers:
        out_response.headers.add("Access-Control-Allow-Origin", in_request.headers['Origin'])
    else:
        out_response.headers.add("Access-Control-Allow-Origin", in_request.host_url)

    return out_response


@flask_app.route("/")
def hello():
    return render_template('index.html')


@flask_app.route('/health', methods=['GET'])
def health():
    print('/health')
    logging.info('/health')
    response = make_response('ok')
    return add_cors_to_response(request, response)


@flask_app.route('/perform_image', methods=['POST', 'OPTIONS'])
def run():
    data = request.data
    # headers = request.headers
    start_time = time.time()
    try:
        j = json.loads(data)
        image_base64 = j['image_base64']
    except:
        json_dict = {"Status": "Error", "Results": "Json parsing error"}
        logging.error('Json parsing error')
        response = make_response(json.dumps(json_dict))
        return add_cors_to_response(request, response)

    try:
        raw_image_bytes = base64.b64decode(image_base64)
        file_bytes = np.asarray(bytearray(raw_image_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except:
        json_dict = {"Status": "Error", "Results": "Jpeg decoding error"}
        logging.error('Jpeg decoding error')
        response = make_response(json.dumps(json_dict))
        return add_cors_to_response(request, response)

    try:
        result_image, output_json = generate_images(img, app, mapper, pipeline)
    except:
        json_dict = {"Status": "Error", "Results": "Processing error"}
        logging.error('Processing error')
        response = make_response(json.dumps(json_dict))
        return add_cors_to_response(request, response)

    image_for_send = result_image
    _, image_bytes = cv2.imencode('.jpg', image_for_send)
    base64_bytes = base64.b64encode(image_bytes)
    base64_string = base64_bytes.decode('utf-8')

    json_dict = {"Status": "Success", "Results": output_json, "Image_base64": base64_string}
    str_json = json.dumps(json_dict)

    logging.info(f'/image_perform with time:  {time.time() - start_time}')
    print(f"Image successfully performed. Time: {time.time() - start_time}\n")

    response = make_response(str_json)
    return add_cors_to_response(request, response)


flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app)
if __name__ == '__main__':
    # test web service
    # port = int(os.environ.get('PORT', 7000))
    # flask_app.run(debug=True, host='0.0.0.0', port=port)

    # prod web service
    http_server = WSGIServer(flask_app, host='0.0.0.0', port=7000)
    http_server.start()

