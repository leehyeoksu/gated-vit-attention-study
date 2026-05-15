import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


BG = "#12121f"
TEXT_COLOR = "white"


def _to_2d(arr, num_patches):
    try:
        return arr.reshape(num_patches, num_patches)
    except ValueError:
        n = int(np.sqrt(len(arr)))
        return arr[: n * n].reshape(n, n)


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img

    if hasattr(img, "detach"):
        arr = img.detach().cpu().permute(1, 2, 0).numpy()
    else:
        arr = np.transpose(img, (1, 2, 0)) if img.ndim == 3 and img.shape[0] in [1, 3] else img

    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _attention_map(raw, num_patches, orig_size, top_percent=5.0):
    percentile = 100 - top_percent
    thr = np.percentile(raw, percentile)
    data = np.where(raw >= thr, raw, 0.0)
    mx = data.max()
    if mx > 1e-8:
        data = data / mx
    arr_2d = _to_2d(data, num_patches)
    pil = Image.fromarray((arr_2d * 255).astype(np.uint8))
    pil = pil.resize(orig_size, Image.NEAREST)
    return np.array(pil, dtype=np.float32) / 255.0


def _entropy(prob: np.ndarray) -> float:
    eps = 1e-8
    return float(-np.sum(prob * np.log(prob + eps)))


HEAD_POS = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]


def visualize(
    qk_maps: dict,
    original_img,
    save_dir: str = "outputs",
    patch_size: int = 16,
    img_size: int = 224,
    top_percent: float = 10.0,
    layers=None,
):
    os.makedirs(save_dir, exist_ok=True)

    num_patches = img_size // patch_size
    orig = _to_pil(original_img).resize((img_size, img_size))
    orig_arr = np.array(orig)

    all_layers = sorted(qk_maps.keys())
    target_layers = all_layers if layers is None else [l for l in layers if l in qk_maps]

    for layer_idx in target_layers:
        floor_n = layer_idx + 1
        attn = qk_maps[layer_idx][0]   # (H, N, N)
        num_heads = attn.shape[0]

        entropies = []
        max_attns = []
        sink_scores = []

        for h in range(num_heads):
            cls_attn = attn[h, 0, 1:].detach().cpu().numpy()
            patch_to_cls = attn[h, 1:, 0].detach().cpu().numpy()

            entropies.append(_entropy(cls_attn))
            max_attns.append(float(cls_attn.max()))
            sink_scores.append(float(patch_to_cls.mean()))

        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        fig.patch.set_facecolor(BG)

        for ax in axes.flat:
            ax.set_facecolor(BG)
            ax.axis("off")

        axes[0, 0].imshow(orig_arr, interpolation="nearest")
        axes[0, 0].set_title(
            f"Floor {floor_n} Original",
            color=TEXT_COLOR, fontsize=11, fontweight="bold"
        )

        ax_stats = axes[1, 0]
        stats_lines = [
            f"Floor {floor_n} — Layer Stats",
            "",
            f"Entropy   (avg): {np.mean(entropies):.3f}",
            f"Max Attn  (avg): {np.mean(max_attns):.4f}",
            f"Sink Score(avg): {np.mean(sink_scores):.4f}",
            "",
            "Head별 Entropy:",
        ] + [f"  Head {h+1}: {entropies[h]:.3f}" for h in range(num_heads)]

        ax_stats.text(
            0.05, 0.95, "\n".join(stats_lines),
            transform=ax_stats.transAxes,
            color=TEXT_COLOR,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(facecolor="#1a1a2e", edgecolor="#444", boxstyle="round,pad=0.5")
        )

        for h in range(min(num_heads, len(HEAD_POS))):
            row, col = HEAD_POS[h]
            ax = axes[row, col]

            cls_attn = attn[h, 0, 1:].detach().cpu().numpy()
            attn_up = _attention_map(cls_attn, num_patches, orig.size, top_percent)

            ax.imshow(orig_arr, interpolation="nearest")
            ax.imshow(attn_up, cmap="jet", alpha=0.7, vmin=0, vmax=1, interpolation="nearest")

            ent = entropies[h]
            mxa = max_attns[h]

            ent_max = float(np.log(num_patches ** 2))
            ent_color = "#ff6b6b" if ent < ent_max * 0.5 else "#69db7c"

            ax.set_title(
                f"Head {h + 1}\nH={ent:.3f}  max={mxa:.4f}",
                color=ent_color, fontsize=10, fontweight="bold"
            )

        fig.suptitle(
            f"Floor {floor_n} (Encoder Layer {layer_idx}) — Baseline CLS Attention\n"
            f"DeiT-S/16 pretrained (gate 없음)",
            color=TEXT_COLOR, fontsize=12, fontweight="bold", y=1.01
        )
        plt.tight_layout(pad=0.6)

        out_path = os.path.join(save_dir, f"floor{floor_n:02d}_overview.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"저장: {out_path}")

    print(f"\n완료! {len(target_layers)}개 레이어 → {save_dir}/")