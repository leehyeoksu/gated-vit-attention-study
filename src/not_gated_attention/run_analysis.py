import torch
from .hooks import register_hooks, remove_hooks
from .visualize import visualize

# preprocess는 attention_deep과 공용
from src.attention_deep.preprocess import preprocess, preprocess_pil


def run(
    model,
    image_path: str = None,
    pil_img=None,
    save_dir: str = "outputs",
    device: str = None,
    patch_size: int = 16,
    img_size: int = 224,
    top_percent: float = 10.0,
    layers=None,
):
    """
    Baseline (gate 없는) DeiT-S/16 pretrained attention map 추출 + 시각화
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    if pil_img is not None:
        img_tensor, original_img = preprocess_pil(pil_img, img_size)
    elif image_path is not None:
        img_tensor, original_img = preprocess(image_path, img_size)
    else:
        raise ValueError("image_path 또는 pil_img 중 하나는 필요합니다.")

    img_tensor = img_tensor.to(device)

    qk_maps, hooks = register_hooks(model)
    with torch.no_grad():
        _ = model(img_tensor)
    remove_hooks(hooks)

    print(f"QK maps: {len(qk_maps)}개 레이어")

    visualize(
        qk_maps,
        original_img,
        save_dir=save_dir,
        patch_size=patch_size,
        img_size=img_size,
        top_percent=top_percent,
        layers=layers,
    )

    return qk_maps