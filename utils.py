from dataclasses import dataclass

import cv2
import numpy as np
from scipy.stats import kurtosis, skew


@dataclass
class ImageEmbInfo:
    filename: str
    embedding: np.ndarray


@dataclass
class Stats:
    kurtosis: float
    skew: float
    mean: float
    var: float
    width: float

    def __repr__(self):
        return f'kurt: {self.kurtosis}\nskew: {self.skew}\nmean: {self.mean}\nvar: {self.var}\nmax - min: {self.width}'


def get_embeddings_info(file_path: str, model) -> ImageEmbInfo:
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = model.get(img)
    face = faces[0]
    emb = face.normed_embedding
    return ImageEmbInfo(filename=file_path, embedding=emb)


def get_stats(embedding: np.ndarray) -> Stats:
    kurt = kurtosis(embedding)
    sk = skew(embedding)
    mean = np.mean(embedding)
    w = max(embedding) - min(embedding)
    v = np.var(embedding)
    return Stats(
        kurtosis=kurt,
        skew=sk,
        mean=mean,
        width=w,
        var=v
    )
