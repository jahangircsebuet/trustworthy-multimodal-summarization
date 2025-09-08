import torch
import open_clip
from PIL import Image
import requests
from io import BytesIO

_CLIP_MODEL = None
_CLIP_PREP = None
_TOKENIZER = None

def load_clip():
    global _CLIP_MODEL, _CLIP_PREP, _TOKENIZER
    if _CLIP_MODEL is None:
        _CLIP_MODEL, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        _CLIP_MODEL = _CLIP_MODEL.to("cuda" if torch.cuda.is_available() else "cpu")
        _CLIP_PREP = preprocess
        _TOKENIZER = open_clip.get_tokenizer("ViT-B-32")
    return _CLIP_MODEL, _CLIP_PREP, _TOKENIZER

def load_image(image_path: str):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    else:
        return Image.open(image_path)

def clip_similarity(image_path: str, text: str) -> float:
    model, preprocess, tok = load_clip()
    image = preprocess(load_image(image_path)).unsqueeze(0).to(model.logit_scale.device)
    text_tok = tok([text])
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tok.to(model.logit_scale.device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        sim = (image_features @ text_features.T).item()