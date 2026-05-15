import torch


def _get_backbone(model):
    return model.backbone if hasattr(model, "backbone") else model


def _get_layers_and_config(model):
    backbone = _get_backbone(model)

    # HF DeiT / ViT
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer, backbone.config

    # fallback
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


def register_hooks(model):
    """
    Baseline (gate 없는) DeiT-S/16 pretrained용 attention hook
    QK softmax attention만 캐싱
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
                    # HF DeiT / ViT attention
                    if hasattr(attn_module, "query") and hasattr(attn_module, "key"):
                        Q = attn_module.query(hidden_states)
                        K = attn_module.key(hidden_states)
                    # fallback
                    elif hasattr(attn_module, "q_proj") and hasattr(attn_module, "k_proj"):
                        Q = attn_module.q_proj(hidden_states)
                        K = attn_module.k_proj(hidden_states)
                    else:
                        raise AttributeError("query/key projection을 찾을 수 없습니다.")

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