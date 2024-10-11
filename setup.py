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


# clone_repo('https://github.com/foivospar/Arc2Face.git', "./arc2face")

# subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "./arc2face/requirements.txt"])
# subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "huggingface_hub"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub==0.23.0"])

from huggingface_hub import hf_hub_download

# hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/config.json", local_dir="./models")
# hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/diffusion_pytorch_model.safetensors", local_dir="./models")
# hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/config.json", local_dir="./models")
# hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/pytorch_model.bin", local_dir="./models")
# hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arcface.onnx", local_dir="./models/antelopev2")

from insightface.utils.storage import download_onnx

download_onnx('models', "buffalo_l", root="./models/buffalo_l", download_zip=True)

move_files_and_remove_folder("./models/buffalo_l/models", "./models/buffalo_l")
