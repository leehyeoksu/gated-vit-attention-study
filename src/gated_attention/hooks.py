import torch


def _get_backbone(model):
    return model.backbone if hasattr(model, "backbone") else model


def _get_layers_and_config(model):
    backbone = _get_backbone(model)

    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer, backbone.config

    if hasattr(backbone, "layer"):
        return backbone.layer, backbone.config

    if hasattr(backbone, "blocks"):
        return backbone.blocks, backbone.config

    raise AttributeError("지원되지 않는 backbone 구조입니다.")


def _get_attn_core(layer):
    # HF DeiT / ViT
    if hasattr(layer, "attention") and hasattr(layer.attention, "attention"):
        return layer.attention.attention

    # fallback
    if hasattr(layer, "attention"):
        return layer.attention

    raise AttributeError("attention core를 찾을 수 없습니다.")


def _get_output_proj(layer):
    # HF DeiT / ViT
    if hasattr(layer, "attention") and hasattr(layer.attention, "output"):
        return layer.attention.output.dense

    # DINO-style fallback
    if hasattr(layer, "attention") and hasattr(layer.attention, "o_proj"):
        return layer.attention.o_proj

    raise AttributeError("attention output projection을 찾을 수 없습니다.")


def register_hooks(model):
    """
    DeiT-S/16 pretrained 기준 attention hook
    실제 attention core의 q/k로 QK softmax 계산

    Returns:
        qk_maps : {layer_idx: tensor(B, H, N, N)}
        hooks   : hook handle 리스트
    """
    layers, config = _get_layers_and_config(model)
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    qk_maps = {}
    hooks = []

    for i, layer in enumerate(layers):
        attn_core = _get_attn_core(layer)

        def make_hook(idx, attn_module):
            def hook(module, args, kwargs):
                hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                if hidden_states is None:
                    return

                B, N, _ = hidden_states.shape

                with torch.no_grad():
                    Q = attn_module.query(hidden_states)
                    K = attn_module.key(hidden_states)

                    Q = Q.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
                    K = K.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

                    scale = head_dim ** -0.5
                    attn = torch.softmax((Q @ K.transpose(-2, -1)) * scale, dim=-1)
                    qk_maps[idx] = attn.detach().cpu()

            return hook

        h = attn_core.register_forward_pre_hook(
            make_hook(i, attn_core), with_kwargs=True
        )
        hooks.append(h)

    return qk_maps, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def collect_gates(model):
    """
    forward 후 각 레이어 gate 수집
    Returns:
        gate_maps: {layer_idx: tensor(B, N, H, Dh)}
    """
    from src.re_make_Gate import GatedOutputProjection

    layers, _ = _get_layers_and_config(model)
    gate_maps = {}

    for i, layer in enumerate(layers):
        out_proj = _get_output_proj(layer)
        if isinstance(out_proj, GatedOutputProjection) and out_proj.last_gate is not None:
            gate_maps[i] = out_proj.last_gate.detach().cpu()

    return gate_maps