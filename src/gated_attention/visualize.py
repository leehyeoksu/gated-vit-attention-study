import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats as scipy_stats


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


def _gate_map(gate_h, num_patches, orig_size, gate_vmin, gate_vmax):
    arr_2d = _to_2d(gate_h, num_patches)
    span = gate_vmax - gate_vmin + 1e-8
    normed = ((arr_2d - gate_vmin) / span * 255).astype(np.uint8)
    pil = Image.fromarray(normed)
    pil = pil.resize(orig_size, Image.NEAREST)
    return np.array(pil, dtype=np.float32) / 255.0


def _diff_map(diff_raw, num_patches, orig_size):
    vabs = np.abs(diff_raw).max() + 1e-8
    normed = ((diff_raw / vabs + 1.0) * 127.5).astype(np.uint8)
    arr_2d = _to_2d(normed, num_patches)
    pil = Image.fromarray(arr_2d)
    pil = pil.resize(orig_size, Image.NEAREST)
    return np.array(pil, dtype=np.float32) / 255.0


def _make_axes_2d(axes, nrows, ncols):
    axes = np.array(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        if nrows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(-1, 1)
    return axes


def _gate_attn_corr(gate_h: np.ndarray, cls_attn: np.ndarray):
    if np.std(gate_h) < 1e-8 or np.std(cls_attn) < 1e-8:
        return None, None
    pr, _ = scipy_stats.pearsonr(gate_h, cls_attn)
    sr, _ = scipy_stats.spearmanr(gate_h, cls_attn)
    return float(pr), float(sr)


def visualize(
    qk_maps: dict,
    gate_maps: dict,
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

    # DeiT-S/16 head 수는 6이라 기본 2x4로 충분
    HEAD_POS = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]

    for layer_idx in target_layers:
        floor_n = layer_idx + 1
        attn = qk_maps[layer_idx][0]
        num_heads = attn.shape[0]

        has_gate = layer_idx in gate_maps
        gate = gate_maps[layer_idx][0] if has_gate else None

        if has_gate:
            gate_patch_avg = gate[1:, :, :].mean(axis=(1, 2))
            gate_patch_avg = gate_patch_avg.detach().cpu().numpy() if hasattr(gate_patch_avg, "detach") else gate_patch_avg
            gate_vmin = float(gate_patch_avg.min())
            gate_vmax = float(gate_patch_avg.max())
            gate_up_avg = _gate_map(gate_patch_avg, num_patches, orig.size, gate_vmin, gate_vmax)

        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        fig.patch.set_facecolor(BG)

        for ax in axes.flat:
            ax.set_facecolor(BG)
            ax.axis("off")

        ax_gate = axes[0, 0]
        if has_gate:
            im = ax_gate.imshow(gate_up_avg, cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
            ax_gate.set_title(
                f"Floor {floor_n}  Gate\n(all heads avg)",
                color=TEXT_COLOR, fontsize=11, fontweight="bold"
            )
            cbar = plt.colorbar(im, ax=ax_gate, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors="#c8c8e8", labelsize=7)
            mid = (gate_vmin + gate_vmax) / 2
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels([f"{gate_vmin:.3f}", f"{mid:.3f}", f"{gate_vmax:.3f}"])
        else:
            ax_gate.set_title(f"Floor {floor_n}  Gate: 없음", color=TEXT_COLOR, fontsize=11)

        axes[1, 0].imshow(orig_arr, interpolation="nearest")
        axes[1, 0].set_title("Original", color=TEXT_COLOR, fontsize=11)

        for h in range(min(num_heads, len(HEAD_POS))):
            row, col = HEAD_POS[h]
            ax = axes[row, col]

            cls_attn = attn[h, 0, 1:].detach().cpu().numpy()
            attn_up = _attention_map(cls_attn, num_patches, orig.size, top_percent)

            ax.imshow(orig_arr, interpolation="nearest")
            ax.imshow(attn_up, cmap="jet", alpha=0.7, vmin=0, vmax=1, interpolation="nearest")

            if has_gate:
                gate_h = gate[1:, h, :].mean(axis=-1)
                gate_h = gate_h.detach().cpu().numpy() if hasattr(gate_h, "detach") else gate_h
                pr, sr = _gate_attn_corr(gate_h, cls_attn)
                if pr is not None:
                    corr_color = "#ff6b6b" if pr > 0 else "#69db7c"
                    corr_str = f"r={pr:+.3f}  ρ={sr:+.3f}"
                else:
                    corr_color = TEXT_COLOR
                    corr_str = "r=N/A"

                ax.set_title(
                    f"Head {h + 1}\n{corr_str}",
                    color=corr_color, fontsize=10, fontweight="bold"
                )
            else:
                ax.set_title(f"Head {h + 1}", color=TEXT_COLOR, fontsize=10)

        fig.suptitle(
            f"Floor {floor_n} (Encoder Layer {layer_idx}) — Gate & CLS Attention\n"
            f"DeiT-S/16 pretrained | r=Pearson  ρ=Spearman",
            color=TEXT_COLOR, fontsize=12, fontweight="bold", y=1.01
        )
        plt.tight_layout(pad=0.6)

        out_path = os.path.join(save_dir, f"floor{floor_n:02d}_overview.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"저장: {out_path}")

    print(f"\n완료! {len(target_layers)}개 레이어 → {save_dir}/")


def visualize_comparison(
    baseline_qk: dict,
    gated_qk: dict,
    gate_maps: dict,
    original_img,
    save_dir: str = "outputs",
    patch_size: int = 16,
    img_size: int = 224,
    top_percent: float = 10.0,
    layers=None,
):
    comp_dir = os.path.join(save_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    num_patches = img_size // patch_size
    orig = _to_pil(original_img).resize((img_size, img_size))

    all_layers = sorted(set(baseline_qk.keys()) & set(gated_qk.keys()))
    if layers is not None:
        all_layers = [l for l in all_layers if l in layers]

    for layer_idx in all_layers:
        base_attn = baseline_qk[layer_idx][0]
        gate_attn = gated_qk[layer_idx][0]
        H = base_attn.shape[0]

        has_gate = layer_idx in gate_maps
        gate = gate_maps[layer_idx][0] if has_gate else None

        if has_gate:
            gate_patch = gate[1:, :, :].mean(axis=-1)
            gate_patch = gate_patch.detach().cpu().numpy() if hasattr(gate_patch, "detach") else gate_patch
            gate_vmin = float(gate_patch.min())
            gate_vmax = float(gate_patch.max())

        nrows = 4 if has_gate else 3
        ncols = H

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
        fig.patch.set_facecolor(BG)
        axes = _make_axes_2d(axes, nrows, ncols)

        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor(BG)

        row_labels = ["Baseline Attn", "Gated Attn", "Diff (Gated−Base)"]
        if has_gate:
            row_labels.append("Gate Heatmap")

        for h in range(H):
            raw_base = base_attn[h, 0, 1:].detach().cpu().numpy()
            raw_gated = gate_attn[h, 0, 1:].detach().cpu().numpy()

            attn_base = _attention_map(raw_base, num_patches, orig.size, top_percent)
            attn_gated = _attention_map(raw_gated, num_patches, orig.size, top_percent)
            diff_map = _diff_map(raw_gated - raw_base, num_patches, orig.size)

            ax = axes[0][h]
            ax.imshow(orig, interpolation="nearest")
            ax.imshow(attn_base, cmap="jet", alpha=0.7, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"H{h}", color=TEXT_COLOR, fontsize=9, fontweight="bold")
            ax.axis("off")

            ax = axes[1][h]
            ax.imshow(orig, interpolation="nearest")
            ax.imshow(attn_gated, cmap="jet", alpha=0.7, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"H{h}", color=TEXT_COLOR, fontsize=9, fontweight="bold")
            ax.axis("off")

            ax = axes[2][h]
            ax.imshow(orig, interpolation="nearest")
            ax.imshow(diff_map, cmap="RdBu_r", alpha=0.8, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"H{h}", color=TEXT_COLOR, fontsize=9, fontweight="bold")
            ax.axis("off")

            if has_gate:
                gate_h = gate[1:, h, :].mean(axis=-1)
                gate_h = gate_h.detach().cpu().numpy() if hasattr(gate_h, "detach") else gate_h
                gate_up = _gate_map(gate_h, num_patches, orig.size, gate_vmin, gate_vmax)
                ax = axes[3][h]
                ax.imshow(gate_up, cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"H{h}", color=TEXT_COLOR, fontsize=9, fontweight="bold")
                ax.axis("off")

        for row_i, label in enumerate(row_labels):
            axes[row_i][0].set_ylabel(label, color=TEXT_COLOR, fontsize=10)

        fig.suptitle(
            f"Layer {layer_idx} — Baseline vs Gated Attention (DeiT-S/16 pretrained)",
            color=TEXT_COLOR, fontsize=13, fontweight="bold", y=1.01
        )
        plt.tight_layout(pad=0.5)

        out_path = os.path.join(comp_dir, f"layer_{layer_idx:02d}_comparison.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"저장: {out_path}")

    print(f"\n비교 시각화 완료 → {comp_dir}/")


def plot_metrics_comparison(base_metrics: dict, gated_metrics: dict, save_dir: str = "outputs"):
    comp_dir = os.path.join(save_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    layers = sorted(set(base_metrics.keys()) & set(gated_metrics.keys()))

    metrics_cfg = [
        # (key, ylabel, note, 낮을수록 좋은지)
        ("entropy",            "Entropy",            "↑ = 분산, sink 감소",        False),
        ("sink_score",         "Sink Score",          "↓ = sink 감소 ✅",           True),
        ("max_attn",           "Max Attention",       "↓ = sink 감소 ✅",           True),
        ("gini",               "Gini Coefficient",    "↓ = 균등, sink 감소 ✅",     True),
        ("cls_received_ratio", "CLS Recv Ratio",      "↓ = CLS sink 감소 ✅",      True),
        ("top1_ratio",         "Top-1 Ratio",         "↓ = 집중 감소 ✅",           True),
        ("effective_rank",     "Effective Rank",      "↑ = 분산, sink 감소 ✅",     False),
    ]

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 10))
    fig.patch.set_facecolor(BG)

    axes_flat = axes.flat
    for ax, (key, ylabel, note, lower_better) in zip(axes_flat, metrics_cfg):
        ax.set_facecolor("#1a1a2e")

        base_vals  = [base_metrics[l][key]  for l in layers]
        gated_vals = [gated_metrics[l][key] for l in layers]

        c_base  = "#7ec8e3"
        c_gated = "#ff9f43" if lower_better else "#55efc4"

        ax.plot(layers, base_vals,  "o-", linewidth=2, markersize=6,
                color=c_base,  label="Baseline")
        ax.plot(layers, gated_vals, "s-", linewidth=2, markersize=6,
                color=c_gated, label="Gated")
        ax.fill_between(layers, base_vals, gated_vals, alpha=0.12, color=c_gated)

        ax.set_xlabel("Layer", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel(ylabel,  color=TEXT_COLOR, fontsize=9)
        ax.set_title(f"{ylabel}\n{note}", color=TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.legend(facecolor=BG, labelcolor=TEXT_COLOR, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # 빈 마지막 축 숨기기
    for ax in list(axes_flat):
        ax.set_visible(False)

    fig.suptitle(
        "Baseline vs Gated — Attention Sink Metrics per Layer\n"
        "(Sink Score / Gini / CLS Recv / Top-1: ↓ better  |  Entropy / Eff. Rank: ↑ better)",
        color=TEXT_COLOR, fontsize=12, fontweight="bold"
    )
    plt.tight_layout(pad=1.2)

    out_path = os.path.join(comp_dir, "metrics_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"저장: {out_path}")


def plot_gate_correlation(gate_corr: dict, save_dir: str = "outputs"):
    comp_dir = os.path.join(save_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    layers = sorted(gate_corr.keys())
    if not layers:
        return

    num_heads = len(gate_corr[layers[0]]["pearson_per_head"])

    pearson_mat = np.array([gate_corr[l]["pearson_per_head"] for l in layers])
    spearman_mat = np.array([gate_corr[l]["spearman_per_head"] for l in layers])

    for mat, name, title in [
        (pearson_mat, "pearson", "Pearson r  (Gate vs CLS→Patch Attention)"),
        (spearman_mat, "spearman", "Spearman r (Gate vs CLS→Patch Attention)"),
    ]:
        fig, ax = plt.subplots(figsize=(max(6, num_heads * 1.2), max(4, len(layers) * 0.6)))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)], color=TEXT_COLOR, fontsize=9)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers], color=TEXT_COLOR, fontsize=9)
        ax.set_xlabel("Head", color=TEXT_COLOR)
        ax.set_ylabel("Layer", color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold")

        for i, _ in enumerate(layers):
            for j in range(num_heads):
                ax.text(
                    j, i, f"{mat[i, j]:+.2f}",
                    ha="center", va="center",
                    color="black" if abs(mat[i, j]) < 0.5 else "white",
                    fontsize=7
                )

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        cbar.set_label("correlation", color=TEXT_COLOR, fontsize=9)

        plt.tight_layout(pad=0.5)
        out_path = os.path.join(comp_dir, f"gate_corr_{name}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"저장: {out_path}")