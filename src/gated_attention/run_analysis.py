import torch
from .preprocess import preprocess, preprocess_pil
from .hooks      import register_hooks, remove_hooks, collect_gates
from .visualize  import visualize, visualize_comparison, plot_metrics_comparison, plot_gate_correlation
from .analyze    import analyze, compare_models, compute_metrics


# =========================================================
# 1. 단일 모델 분석 (기존 run 유지)
# =========================================================
def run(
    model,
    image_path: str    = None,
    pil_img            = None,
    save_dir: str      = "outputs",
    device: str        = None,
    patch_size: int    = 8,   
    img_size: int      = 64,  
    top_percent: float = 10.0,
    layers             = None, 
):
    """
    단일 gated 모델 분석 파이프라인
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # ── 전처리
    if pil_img is not None:
        img_tensor, original_img = preprocess_pil(pil_img, img_size)
    elif image_path is not None:
        img_tensor, original_img = preprocess(image_path, img_size)
    else:
        raise ValueError("image_path 또는 pil_img 중 하나는 필요합니다.")

    img_tensor = img_tensor.to(device)

    # ── Hook 등록 → Forward → Hook 제거
    qk_maps, hooks = register_hooks(model)
    with torch.no_grad():
        _ = model(img_tensor)
    remove_hooks(hooks)

    # ── Gate 수집
    gate_maps = collect_gates(model)

    print(f"QK maps  : {len(qk_maps)}개 레이어")
    print(f"Gate maps: {len(gate_maps)}개 레이어")

    # ── 시각화
    visualize(
        qk_maps, gate_maps, original_img,
        save_dir    = save_dir,
        patch_size  = patch_size,
        img_size    = img_size,
        top_percent = top_percent,
        layers      = layers,
    )

    # ── 통계
    results = analyze(qk_maps, gate_maps)

    return qk_maps, gate_maps, results


# =========================================================
# 2. Baseline vs Gated 비교 분석 파이프라인
# =========================================================
def run_comparison(
    baseline_model,
    gated_model,
    image_path: str    = None,
    pil_img            = None,
    save_dir: str      = "outputs",
    device: str        = None,
    patch_size: int    = 8,   
    img_size: int      = 64,  
    top_percent: float = 10.0,
    layers             = None, 
):
    """
    Baseline vs Gated 비교 분석 파이프라인
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 전처리 (두 모델 공용)
    if pil_img is not None:
        img_tensor, original_img = preprocess_pil(pil_img, img_size)
    elif image_path is not None:
        img_tensor, original_img = preprocess(image_path, img_size)
    else:
        raise ValueError("image_path 또는 pil_img 중 하나는 필요합니다.")

    img_tensor = img_tensor.to(device)

    # ── Baseline forward
    print("[1/4] Baseline 모델 forward...")
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    baseline_qk, base_hooks = register_hooks(baseline_model)
    with torch.no_grad():
        _ = baseline_model(img_tensor)
    remove_hooks(base_hooks)
    print(f"  → QK maps: {len(baseline_qk)}개 레이어")

    # ── Gated forward
    print("[2/4] Gated 모델 forward...")
    gated_model = gated_model.to(device)
    gated_model.eval()
    gated_qk, gated_hooks = register_hooks(gated_model)
    with torch.no_grad():
        _ = gated_model(img_tensor)
    remove_hooks(gated_hooks)
    gate_maps = collect_gates(gated_model)
    print(f"  → QK maps: {len(gated_qk)}개 레이어  |  Gate maps: {len(gate_maps)}개 레이어")

    # ── 비교 시각화
    print("[3/4] 비교 시각화 저장 중...")
    visualize_comparison(
        baseline_qk, gated_qk, gate_maps, original_img,
        save_dir    = save_dir,
        patch_size  = patch_size,
        img_size    = img_size,
        top_percent = top_percent,
        layers      = layers,
    )

    # ── 메트릭 차트
    base_metrics  = compute_metrics(baseline_qk)
    gated_metrics = compute_metrics(gated_qk)
    plot_metrics_comparison(base_metrics, gated_metrics, save_dir=save_dir)

    # ── 통계 분석
    print("[4/4] 통계 분석 중...")
    results = compare_models(baseline_qk, gated_qk, gate_maps)

    # ── Gate 상관관계 히트맵
    if results.get("gate_corr"):
        plot_gate_correlation(results["gate_corr"], save_dir=save_dir)

    print(f"\n모든 결과 저장 완료 → {save_dir}/")
    return baseline_qk, gated_qk, gate_maps, results