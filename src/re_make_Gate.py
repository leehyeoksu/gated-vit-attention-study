import torch
import torch.nn as nn


def _get_backbone(model):
    if hasattr(model, "backbone"):
        return model.backbone
    return model


def _get_layers(backbone):
    # HuggingFace ViT/DeiT
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer

    # 혹시 다른 구현 대비
    if hasattr(backbone, "layer"):
        return backbone.layer

    if hasattr(backbone, "blocks"):
        return backbone.blocks

    raise AttributeError("지원되지 않는 backbone 구조입니다. encoder.layer / layer / blocks 없음.")


def _get_attn_and_output(layer):
    """
    HF ViT/DeiT 기준:
      - self-attention: layer.attention.attention
      - output proj   : layer.attention.output.dense
    """
    if hasattr(layer, "attention"):
        # HF ViT/DeiT
        if hasattr(layer.attention, "attention") and hasattr(layer.attention, "output"):
            attn_core = layer.attention.attention
            out_dense = layer.attention.output.dense
            return attn_core, out_dense, "hf_vit"

        # DINOv3식 fallback
        if hasattr(layer.attention, "o_proj"):
            attn_core = layer.attention
            out_dense = layer.attention.o_proj
            return attn_core, out_dense, "dino_style"

    raise AttributeError("지원되지 않는 attention 구조입니다.")


def _set_output_proj(layer, new_module, style):
    if style == "hf_vit":
        layer.attention.output.dense = new_module
    elif style == "dino_style":
        layer.attention.o_proj = new_module
    else:
        raise ValueError(f"알 수 없는 style: {style}")


class GatedOutputProjection(nn.Module):
    """
    gate 위치:
      - attention output dense 직전
      - 입력 x (B, N, d_model)에 sigmoid gate 적용 후 원래 projection 통과
    """
    def __init__(self, original_proj, d_model, num_heads, keep_cls_ungated=True):
        super().__init__()

        self.original_proj     = original_proj
        self.d_model           = d_model
        self.num_heads         = num_heads
        self.head_dim          = d_model // num_heads
        self.keep_cls_ungated  = keep_cls_ungated

        self.gate_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.01)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self.cached_hidden_states = None
        self.last_gate = None

    def set_hidden_states(self, hidden_states):
        self.cached_hidden_states = hidden_states

    def forward(self, x):
        # x: (B, N, d_model)
        if self.cached_hidden_states is None:
            return self.original_proj(x)

        hs = self.cached_hidden_states
        B, N, _ = hs.shape

        g = torch.sigmoid(self.gate_proj(hs))            # (B, N, d_model)
        g = g.view(B, N, self.num_heads, self.head_dim)  # (B, N, H, Dh)

        if self.keep_cls_ungated and N > 0:
            g = g.clone()
            g[:, 0] = 1.0

        self.last_gate = g

        x = x.view(B, N, self.num_heads, self.head_dim)
        x = g * x
        x = x.reshape(B, N, self.d_model)

        out = self.original_proj(x)
        self.cached_hidden_states = None
        return out


def _make_cache_hook(wrapper):
    def hook(module, args, kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states", None)
        if hidden_states is not None:
            wrapper.set_hidden_states(hidden_states)
        return args, kwargs
    return hook


def inject_gating(model, keep_cls_ungated=True, target_layers=None):
    """
    DeiT / ViT / 일부 DINO-style 공용

    target_layers:
      - None        : 전체 레이어
      - int         : 마지막 n개 레이어
      - list/tuple  : 지정 레이어 인덱스
    """
    backbone = _get_backbone(model)
    layers = _get_layers(backbone)

    d_model = backbone.config.hidden_size
    num_heads = backbone.config.num_attention_heads
    total = len(layers)

    if target_layers is None:
        indices = list(range(total))
    elif isinstance(target_layers, int):
        indices = list(range(total - target_layers, total))
    else:
        indices = list(target_layers)

    hooks = []

    for i, layer in enumerate(layers):
        if i not in indices:
            continue

        attn_core, out_proj, style = _get_attn_and_output(layer)

        wrapper = GatedOutputProjection(
            original_proj=out_proj,
            d_model=d_model,
            num_heads=num_heads,
            keep_cls_ungated=keep_cls_ungated,
        )

        handle = attn_core.register_forward_pre_hook(
            _make_cache_hook(wrapper), with_kwargs=True
        )
        hooks.append(handle)

        _set_output_proj(layer, wrapper, style)

    print(
        f"[inject_gating] {total}개 레이어 중 {len(indices)}개에 gate 적용 "
        f"(keep_cls_ungated={keep_cls_ungated})"
    )
    return model, hooks


def compute_gate_loss(model, lambda_l1=0.005, lambda_entropy=0.003):
    backbone = _get_backbone(model)
    layers = _get_layers(backbone)

    l1_total = 0.0
    ent_total = 0.0
    count = 0

    for layer in layers:
        try:
            _, out_proj, _ = _get_attn_and_output(layer)
        except Exception:
            continue

        if not isinstance(out_proj, GatedOutputProjection):
            continue

        g = out_proj.last_gate
        if g is None:
            continue

        patch_gate = g[:, 1:].to(torch.float32)  # CLS 제외

        l1 = patch_gate.mean()
        eps = 1e-8
        H = -(patch_gate * torch.log(patch_gate + eps)
              + (1.0 - patch_gate) * torch.log(1.0 - patch_gate + eps))
        ent = -H.mean()

        l1_total += l1
        ent_total += ent
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    return lambda_l1 * (l1_total / count) + lambda_entropy * (ent_total / count)


def freeze_except_gate(model):
    for p in model.parameters():
        p.requires_grad = False

    backbone = _get_backbone(model)
    layers = _get_layers(backbone)

    for layer in layers:
        try:
            _, out_proj, _ = _get_attn_and_output(layer)
        except Exception:
            continue

        if isinstance(out_proj, GatedOutputProjection):
            for p in out_proj.gate_proj.parameters():
                p.requires_grad = True

    print("[freeze_except_gate] gate_proj만 학습 가능")