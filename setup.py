import subprocess
import sys
import os
import shutil
from pathlib import Path


def move_files_and_remove_folder(source_folder, destination_folder):
    source = Path(source_folder)
    destination = Path(destination_folder)

    for file in source.iterdir():
        if file.is_file():
            shutil.move(str(file), str(destination))  # Перемещение файла

    source.rmdir()


subprocess.check_call(['git', 'clone', 'https://github.com/foivospar/Arc2Face.git', "./arc2face"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0", "--index-url", "https://download.pytorch.org/whl/cu124"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub==0.24.0", "opencv-python"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "./arc2face/requirements.txt"])


from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/config.json", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/diffusion_pytorch_model.safetensors", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/config.json", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/pytorch_model.bin", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arcface.onnx", local_dir="./models/antelopev2")

from insightface.utils.storage import download_onnx

download_onnx('models', "buffalo_l", root="./models/buffalo_l", download_zip=True)

move_files_and_remove_folder("./models/buffalo_l/models", "./models/buffalo_l")
