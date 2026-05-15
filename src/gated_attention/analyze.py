import numpy as np
from scipy import stats


def _to_numpy(t):
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def compute_entropy(attn_vec: np.ndarray) -> float:
    p = np.asarray(attn_vec, dtype=np.float64)
    eps = 1e-8
    return float(-np.sum(p * np.log(p + eps)))


def compute_max_attention(attn_vec: np.ndarray) -> float:
    return float(np.max(attn_vec))


def compute_cls_concentration(attn_vec: np.ndarray, top_k: int = 5) -> float:
    p = np.asarray(attn_vec)
    top_k = min(top_k, len(p))
    return float(np.sort(p)[::-1][:top_k].sum())


def compute_attention_variance(attn_vec: np.ndarray) -> float:
    return float(np.var(attn_vec))


def compute_attention_sink_score(attn_vec: np.ndarray) -> float:
    p = np.asarray(attn_vec)
    mean = p.mean()
    if mean < 1e-8:
        return 0.0
    return float(p.max() / mean)


def compute_gini(attn_vec: np.ndarray) -> float:
    """Gini coefficient — 0=완전균등, 1=완전집중 (sink 심할수록 ↑)"""
    p = np.asarray(attn_vec, dtype=np.float64)
    p = np.sort(p)
    n = len(p)
    if n == 0 or p.sum() < 1e-8:
        return 0.0
    cumsum = np.cumsum(p)
    return float((2 * np.sum(cumsum) - (n + 1) * p.sum()) / (n * p.sum()))


def compute_cls_received_ratio(attn_matrix_head: np.ndarray) -> float:
    """패치 → CLS 방향 attention 합산 비율 (열 0 합 / 전체 합)
    attn_matrix_head: (N, N) — 행=query, 열=key
    patch→CLS : attn_matrix_head[1:, 0]
    CLS sink 심할수록 ↑
    """
    a = np.asarray(attn_matrix_head, dtype=np.float64)
    total = a.sum()
    if total < 1e-8:
        return 0.0
    patch_to_cls = a[1:, 0].sum()  # 패치 쿼리들이 CLS 키에 보내는 attention
    return float(patch_to_cls / total)


def compute_top1_ratio(attn_vec: np.ndarray) -> float:
    """CLS→Patch attention 중 최대값 토큰이 차지하는 비율 (↑ → sink 심함)"""
    p = np.asarray(attn_vec, dtype=np.float64)
    total = p.sum()
    if total < 1e-8:
        return 0.0
    return float(p.max() / total)


def compute_effective_rank(attn_vec: np.ndarray) -> float:
    """exp(entropy) / N — 1에 가까울수록 균등분포, 0에 가까울수록 sink 심함"""
    p = np.asarray(attn_vec, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0
    eps = 1e-8
    h = -np.sum(p * np.log(p + eps))
    return float(np.exp(h) / n)


def compute_covariance(gate_vec: np.ndarray, attn_vec: np.ndarray) -> float:
    g, a = np.asarray(gate_vec), np.asarray(attn_vec)
    return float(np.cov(g, a)[0, 1])


def compute_pearson(gate_vec: np.ndarray, attn_vec: np.ndarray) -> float:
    g, a = np.asarray(gate_vec), np.asarray(attn_vec)
    if np.std(g) < 1e-8 or np.std(a) < 1e-8:
        return 0.0
    r, _ = stats.pearsonr(g, a)
    return float(r)


def compute_spearman(gate_vec: np.ndarray, attn_vec: np.ndarray) -> float:
    g, a = np.asarray(gate_vec), np.asarray(attn_vec)
    if np.std(g) < 1e-8 or np.std(a) < 1e-8:
        return 0.0
    r, _ = stats.spearmanr(g, a)
    return float(r)


def compute_metrics(qk_maps: dict) -> dict:
    results = {}
    for layer_idx in sorted(qk_maps.keys()):
        attn = _to_numpy(qk_maps[layer_idx][0])  # (H, N, N)
        H = attn.shape[0]

        per_head = {
            k: [] for k in (
                "entropy", "max_attn", "sink_score", "cls_conc", "variance",
                "gini", "cls_received_ratio", "top1_ratio", "effective_rank",
            )
        }

        for h in range(H):
            cls_to_patch = attn[h, 0, 1:]           # CLS→Patch 방향
            attn_head_full = attn[h]                 # (N, N) 전체 행렬

            per_head["entropy"].append(compute_entropy(cls_to_patch))
            per_head["max_attn"].append(compute_max_attention(cls_to_patch))
            per_head["sink_score"].append(compute_attention_sink_score(cls_to_patch))
            per_head["cls_conc"].append(compute_cls_concentration(cls_to_patch))
            per_head["variance"].append(compute_attention_variance(cls_to_patch))
            # ── 신규 지표 ──
            per_head["gini"].append(compute_gini(cls_to_patch))
            per_head["cls_received_ratio"].append(compute_cls_received_ratio(attn_head_full))
            per_head["top1_ratio"].append(compute_top1_ratio(cls_to_patch))
            per_head["effective_rank"].append(compute_effective_rank(cls_to_patch))

        results[layer_idx] = {
            "entropy":            float(np.mean(per_head["entropy"])),
            "max_attn":           float(np.mean(per_head["max_attn"])),
            "sink_score":         float(np.mean(per_head["sink_score"])),
            "cls_conc":           float(np.mean(per_head["cls_conc"])),
            "variance":           float(np.mean(per_head["variance"])),
            "gini":               float(np.mean(per_head["gini"])),
            "cls_received_ratio": float(np.mean(per_head["cls_received_ratio"])),
            "top1_ratio":         float(np.mean(per_head["top1_ratio"])),
            "effective_rank":     float(np.mean(per_head["effective_rank"])),
            "per_head":           per_head,
        }
    return results


def compare_models(baseline_qk: dict, gated_qk: dict, gate_maps: dict) -> dict:
    base_m = compute_metrics(baseline_qk)
    gated_m = compute_metrics(gated_qk)
    layers = sorted(set(base_m) & set(gated_m))

    print("\n" + "=" * 100)
    print("  Baseline vs Gated — Layer-wise Attention Sink 비교")
    print("=" * 100)

    # ── 기존 지표 테이블 ──────────────────────────────────────────
    hdr = (f"{'L':>3} | {'B-Ent':>7} {'G-Ent':>7} {'ΔEnt':>7} | "
           f"{'B-Sink':>7} {'G-Sink':>7} {'ΔSink':>7} | "
           f"{'B-Max':>7} {'G-Max':>7} {'ΔMax':>7}")
    print(hdr)
    print("-" * 80)

    deltas = {
        k: [] for k in (
            "entropy", "sink_score", "max_attn", "cls_conc", "variance",
            "gini", "cls_received_ratio", "top1_ratio", "effective_rank",
        )
    }

    for l in layers:
        for key in deltas:
            deltas[key].append(gated_m[l][key] - base_m[l][key])

        be, ge = base_m[l]["entropy"], gated_m[l]["entropy"]
        bs, gs = base_m[l]["sink_score"], gated_m[l]["sink_score"]
        bm, gm = base_m[l]["max_attn"], gated_m[l]["max_attn"]

        print(
            f"{l:>3} | {be:>7.4f} {ge:>7.4f} {ge-be:>+7.4f} | "
            f"{bs:>7.4f} {gs:>7.4f} {gs-bs:>+7.4f} | "
            f"{bm:>7.4f} {gm:>7.4f} {gm-bm:>+7.4f}"
        )

    # ── 신규 지표 테이블 ──────────────────────────────────────────
    print("\n" + "-" * 100)
    print("  신규 지표 (Gini / CLS Received Ratio / Top-1 Ratio / Effective Rank)")
    print("-" * 100)
    hdr2 = (f"{'L':>3} | {'B-Gini':>7} {'G-Gini':>7} {'ΔGini':>7} | "
            f"{'B-CLS%':>7} {'G-CLS%':>7} {'ΔCLS%':>7} | "
            f"{'B-Top1':>7} {'G-Top1':>7} {'ΔTop1':>7} | "
            f"{'B-EffR':>7} {'G-EffR':>7} {'ΔEffR':>7}")
    print(hdr2)
    print("-" * 100)

    for l in layers:
        bg, gg = base_m[l]["gini"],               gated_m[l]["gini"]
        bc, gc = base_m[l]["cls_received_ratio"],  gated_m[l]["cls_received_ratio"]
        bt, gt = base_m[l]["top1_ratio"],           gated_m[l]["top1_ratio"]
        be2, ge2 = base_m[l]["effective_rank"],    gated_m[l]["effective_rank"]
        print(
            f"{l:>3} | {bg:>7.4f} {gg:>7.4f} {gg-bg:>+7.4f} | "
            f"{bc:>7.4f} {gc:>7.4f} {gc-bc:>+7.4f} | "
            f"{bt:>7.4f} {gt:>7.4f} {gt-bt:>+7.4f} | "
            f"{be2:>7.4f} {ge2:>7.4f} {ge2-be2:>+7.4f}"
        )

    # ── Wilcoxon 검정 ─────────────────────────────────────────────
    print("\n【Wilcoxon 부호순위 검정】 (p<0.05 → 유의미한 변화)")
    _print_wilcoxon(
        "Entropy         ",
        [base_m[l]["entropy"] for l in layers],
        [gated_m[l]["entropy"] for l in layers],
        deltas["entropy"],
        pos_label="분산화",
        neg_label="집중화",
    )
    _print_wilcoxon(
        "Sink Score      ",
        [base_m[l]["sink_score"] for l in layers],
        [gated_m[l]["sink_score"] for l in layers],
        deltas["sink_score"],
        pos_label="Sink 증가",
        neg_label="Sink 감소 ✅",
    )
    _print_wilcoxon(
        "Max Attn        ",
        [base_m[l]["max_attn"] for l in layers],
        [gated_m[l]["max_attn"] for l in layers],
        deltas["max_attn"],
        pos_label="집중 증가",
        neg_label="집중 감소 ✅",
    )
    _print_wilcoxon(
        "Gini            ",
        [base_m[l]["gini"] for l in layers],
        [gated_m[l]["gini"] for l in layers],
        deltas["gini"],
        pos_label="불균등 증가",
        neg_label="균등화 ✅",
    )
    _print_wilcoxon(
        "CLS Recv Ratio  ",
        [base_m[l]["cls_received_ratio"] for l in layers],
        [gated_m[l]["cls_received_ratio"] for l in layers],
        deltas["cls_received_ratio"],
        pos_label="CLS sink 증가",
        neg_label="CLS sink 감소 ✅",
    )
    _print_wilcoxon(
        "Top-1 Ratio     ",
        [base_m[l]["top1_ratio"] for l in layers],
        [gated_m[l]["top1_ratio"] for l in layers],
        deltas["top1_ratio"],
        pos_label="집중 증가",
        neg_label="집중 감소 ✅",
    )
    _print_wilcoxon(
        "Effective Rank  ",
        [base_m[l]["effective_rank"] for l in layers],
        [gated_m[l]["effective_rank"] for l in layers],
        deltas["effective_rank"],
        pos_label="분산화 ✅",
        neg_label="집중화",
    )

    # ── Gate 상관관계 ─────────────────────────────────────────────
    print("\n【Gate vs CLS→Patch Attention 상관관계 (레이어×헤드 평균)】")
    print(f"  {'L':>3} | {'Pearson':>9} {'Spearman':>10} {'Cov':>10}  해석")
    print("  " + "-" * 55)

    gate_corr = {}
    for layer_idx in sorted(gate_maps):
        if layer_idx not in gated_qk:
            continue

        attn = _to_numpy(gated_qk[layer_idx][0])
        gate = _to_numpy(gate_maps[layer_idx][0])
        H = attn.shape[0]

        p_rs, s_rs, covs = [], [], []
        for h in range(H):
            cls_attn = attn[h, 0, 1:]
            gate_h = gate[1:, h, :].mean(axis=-1)
            p_rs.append(compute_pearson(gate_h, cls_attn))
            s_rs.append(compute_spearman(gate_h, cls_attn))
            covs.append(compute_covariance(gate_h, cls_attn))

        mp, ms, mc = np.mean(p_rs), np.mean(s_rs), np.mean(covs)
        direction = "억제 (gate↑ → attn↓) ✅" if mp < 0 else "증폭 (gate↑ → attn↑)"
        print(f"  {layer_idx:>3} | {mp:>+9.4f} {ms:>+10.4f} {mc:>+10.4f}  {direction}")

        gate_corr[layer_idx] = {
            "pearson_per_head":  p_rs,
            "spearman_per_head": s_rs,
            "cov_per_head":      covs,
            "mean_pearson":      float(mp),
            "mean_spearman":     float(ms),
            "mean_cov":          float(mc),
        }

    print("\n" + "=" * 100)

    return {
        "baseline_metrics":    base_m,
        "gated_metrics":       gated_m,
        "delta_entropy":       deltas["entropy"],
        "delta_sink":          deltas["sink_score"],
        "delta_max_attn":      deltas["max_attn"],
        "delta_cls_conc":      deltas["cls_conc"],
        "delta_variance":      deltas["variance"],
        "delta_gini":          deltas["gini"],
        "delta_cls_received":  deltas["cls_received_ratio"],
        "delta_top1_ratio":    deltas["top1_ratio"],
        "delta_effective_rank":deltas["effective_rank"],
        "gate_corr":           gate_corr,
    }


def _print_wilcoxon(label, base_vals, gated_vals, deltas, pos_label, neg_label):
    try:
        stat, p = stats.wilcoxon(base_vals, gated_vals)
        sig = "✅ 유의미" if p < 0.05 else "❌ 유의미하지않음"
        mean_d = float(np.mean(deltas))
        direction = pos_label if mean_d > 0 else neg_label
        print(f"  {label} | stat={stat:.4f}, p={p:.4f}  {sig}  Δ평균={mean_d:+.4f} → {direction}")
    except ValueError as e:
        print(f"  {label} → 계산 불가: {e}")


def analyze(qk_maps: dict, gate_maps: dict) -> dict:
    num_heads = list(qk_maps.values())[0][0].shape[0] if qk_maps else 6

    sink_scores_qk = []
    sink_scores_gated = []
    gate_means = []
    layer_sink_qk = []
    head_gates = {h: [] for h in range(num_heads)}

    for layer_idx in sorted(qk_maps):
        if layer_idx not in gate_maps:
            continue

        attn = _to_numpy(qk_maps[layer_idx][0])
        gate = _to_numpy(gate_maps[layer_idx][0])

        layer_qk, layer_gated, layer_gate = [], [], []

        for h in range(attn.shape[0]):
            p2c = attn[h, 1:, 0]
            sink = p2c.mean()
            gh = gate[1:, h, :].mean(axis=-1)
            gn = gh / (gh.max() + 1e-8)

            layer_qk.append(sink)
            layer_gated.append((p2c * gn).mean())
            layer_gate.append(gh.mean())
            head_gates[h].append(gh.mean())

        sink_scores_qk.append(np.mean(layer_qk))
        sink_scores_gated.append(np.mean(layer_gated))
        gate_means.append(np.mean(layer_gate))
        layer_sink_qk.append(layer_qk)

    print("\n" + "=" * 55)
    print("  통계 분석 결과 (단일 gated 모델)")
    print("=" * 55)

    if len(gate_means) > 1:
        r, p = stats.spearmanr(gate_means, sink_scores_qk)
        sig = "✅ 유의미" if p < 0.05 else "❌ 유의미하지않음"
        print(f"\n① Spearman  (gate평균 vs QK sink score)")
        print(f"   r={r:.4f},  p={p:.4f}  {sig}")
        print("   → 음의 상관: gate↑ → sink↓" if r < 0 else "   → 양의 상관 or 무관")

    try:
        stat, p = stats.wilcoxon(sink_scores_qk, sink_scores_gated)
        sig = "✅ 유의미한 억제" if p < 0.05 else "❌ 유의미하지않음"
        print(f"\n② Wilcoxon  (QK sink vs Gate×QK sink)")
        print(f"   stat={stat:.4f},  p={p:.4f}  {sig}")
    except ValueError as e:
        print(f"\n② Wilcoxon  → 계산 불가: {e}")

    try:
        stat, p = stats.kruskal(*layer_sink_qk)
        print(f"\n③ Kruskal-Wallis  stat={stat:.4f},  p={p:.4f}  "
              f"{'✅ 레이어간 차이 있음' if p < 0.05 else '❌ 차이 없음'}")
    except ValueError as e:
        print(f"\n③ Kruskal-Wallis → 계산 불가: {e}")

    try:
        fd = [head_gates[h] for h in range(num_heads)]
        stat, p = stats.friedmanchisquare(*fd)
        print(f"\n④ Friedman  stat={stat:.4f},  p={p:.4f}  "
              f"{'✅ head간 차이 있음' if p < 0.05 else '❌ 차이 없음'}")
    except ValueError as e:
        print(f"\n④ Friedman → 계산 불가: {e}")

    print("\n" + "=" * 55)
    return {
        "gate_means": gate_means,
        "sink_scores_qk": sink_scores_qk,
        "sink_scores_gated": sink_scores_gated,
    }