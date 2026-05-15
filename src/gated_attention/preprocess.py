from torchvision import transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess(image_path: str, img_size: int = 224):
    """
    DeiT-S/16 pretrained용 전처리
    Returns:
        img_tensor   : (1, 3, H, W) normalized tensor
        original_img : PIL Image (원본 시각화용)
    """
    tf = _build_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0), img


def preprocess_pil(pil_img: Image.Image, img_size: int = 224):
    """
    허깅페이스 데이터셋 등에서 꺼낸 PIL 이미지 전처리
    Returns:
        img_tensor   : (1, 3, H, W) normalized tensor
        original_img : PIL Image (원본 시각화용)
    """
    tf = _build_transform(img_size)
    img = pil_img.convert("RGB")
    return tf(img).unsqueeze(0), img