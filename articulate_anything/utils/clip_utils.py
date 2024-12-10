import torch
import clip
import numpy as np


def cosine_similarity(embedding1, embedding2):
    return (np.dot(embedding1, embedding2)
            / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))


class ClipModel:
    def __init__(self, model_name="ViT-B/32", device=None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

    @torch.inference_mode()
    def get_embedding(self, text):
        if not isinstance(text, list):
            text = [text]
        text = [t.replace("\n", " ") for t in text]
        tokens = clip.tokenize(text).to('cuda')
        return self.model.encode_text(tokens).cpu().numpy()

    def cosine_similarity_text(self, text1, text2):
        embedding1 = np.array(self.get_embedding(text1)).flatten()
        embedding2 = np.array(self.get_embedding(text2)).flatten()
        return cosine_similarity(embedding1, embedding2)
