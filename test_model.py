# test_model.py
# Run: pytest test_model.py
import torch
import torch.nn.functional as F
from PIL import Image
import app as app_module


def make_image(color):
    return Image.new("RGB", (128, 128), color=color)


def test_embedding_is_unit_vector():
    emb = torch.tensor(app_module.embed_image(make_image((100, 100, 100))))
    assert abs(emb.norm().item() - 1.0) < 1e-4


def test_identical_images_have_similarity_near_one():
    img = make_image((80, 60, 40))
    e1 = torch.tensor(app_module.embed_image(img))
    e2 = torch.tensor(app_module.embed_image(img))
    sim = F.cosine_similarity(e1, e2, dim=0).item()
    assert sim > 0.999


def test_visually_different_images_give_different_embeddings():
    e1 = torch.tensor(app_module.embed_image(make_image((10, 10, 10))))
    e2 = torch.tensor(app_module.embed_image(make_image((250, 250, 250))))
    sim = F.cosine_similarity(e1, e2, dim=0).item()
    assert sim < 0.999