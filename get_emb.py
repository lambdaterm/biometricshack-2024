import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics.pairwise import cosine_similarity
import onnx
import cv2
import numpy as np
import onnxruntime
import pandas as pd
from tqdm import tqdm


class ArcFaceONNX:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = 'recognition'
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            # print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            # mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        # print('input mean and std:', self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, providers=['CUDAExecutionProvider'])
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(['CUDAExecutionProvider'])

    # def get(self, img, face):
    #     aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
    #     face.embedding = self.get_feat(aimg).flatten()
    #     return face.embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

embedder_r100 = ArcFaceONNX(r'models/arcface.onnx')
embedder_r50 = ArcFaceONNX(r'models/w600k_r50.onnx')

dataset = ImageFolder(root=r'D:\data\biometrics_hack\casia_webface', transform=np.asarray)

emb_r100 = embedder_r100.get_feat(np.asarray(dataset[0][0]))
emb_r50 = embedder_r50.get_feat(np.asarray(dataset[0][0]))

# Получение списка классов (идентичностей)
classes = dataset.classes  # Список имён классов
class_to_idx = dataset.class_to_idx  # Словарь: имя класса -> индекс

# Получение индексов для каждого класса
class_indices = {}
for idx, (path, label) in enumerate(dataset.samples):
    class_indices.setdefault(label, []).append(idx)

# Выбор случайных 100 идентичностей для тестовой выборки
num_test_classes = 100
test_classes = np.random.choice(list(class_indices.keys()), size=num_test_classes, replace=False)
train_classes = list(set(class_indices.keys()) - set(test_classes))

# Получение индексов для обучающей и тестовой выборок
train_indices = [idx for cls in train_classes for idx in class_indices[cls]]
test_indices = [idx for cls in test_classes for idx in class_indices[cls]]

# Создание обучающей и тестовой выборок
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

def get_embedding(model, img_np):
    # Убедимся, что изображение в формате uint8 и имеет правильные размеры
    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)
    if img_np.shape[2] != 3:
        img_np = np.stack([img_np]*3, axis=-1)  # Если изображение не RGB, повторяем каналы
    emb = model.get_feat(img_np)
    return emb


def prepare_and_save_embeddings(dataset, indices, embedder_r50, embedder_r100, csv_filename):
    data = []
    for idx in tqdm(indices):
        img, label = dataset[idx]
        # Убедимся, что изображение имеет правильную форму и тип
        if img.ndim == 2:  # Если изображение чёрно-белое, конвертируем в RGB
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] != 3:
            img = img[:, :, :3]  # Оставляем только первые 3 канала, если их больше

        emb_r50 = get_embedding(embedder_r50, img)
        emb_r100 = get_embedding(embedder_r100, img)
        data.append({
            'label': label,
            'emb_r50': emb_r50.tolist(),
            'emb_r100': emb_r100.tolist()
        })
    # Конвертируем в DataFrame и сохраняем
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)
    print(f"Embeddings saved to {csv_filename}")

prepare_and_save_embeddings(dataset, test_indices, embedder_r50, embedder_r100, 'test_embeddings.csv')

prepare_and_save_embeddings(dataset, train_indices, embedder_r50, embedder_r100, 'train_embeddings.csv')