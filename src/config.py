# =========================================================
# config.py — 실험 전체 하이퍼파라미터 중앙 관리
# =========================================================

# ── 모델 ──────────────────────────────────────────────────
MODEL_ID    = "facebook/dinov3-vits16plus-pretrain-lvd1689m"  # pretrained DINOv3-S
NUM_CLASSES = 200                        # Tiny ImageNet

# ── 학습 ──────────────────────────────────────────────────
EPOCHS      = 100
PATIENCE    = 3        # early stopping (val loss 기준)
BATCH_SIZE  = 128
LR          = 1e-4
MIN_DELTA   = 1e-4     # early stopping 최소 개선량

# ── 이미지 ────────────────────────────────────────────────
IMG_SIZE    = 224
PATCH_SIZE  = 16       # DINOv3-S patch size

# ── 학습 중 attention map 추출 주기 ───────────────────────
ATTN_INTERVAL = 10     # 매 N epoch마다 attention map 저장 (0=비활성)

# ── Gate loss ─────────────────────────────────────────────
GATE_LAMBDA      = 0.01   # gate_loss 전체 가중치
GATE_LAMBDA_L1   = 0.005  # L1 억제
GATE_LAMBDA_ENT  = 0.003  # Entropy bimodal 유도

# ── 경로 ──────────────────────────────────────────────────
CKPT_DIR     = "outputs/checkpoints"
FIGURES_DIR  = "outputs/figures"
LOGS_DIR     = "outputs/logs"
ATTN_DIR     = "outputs/figures"   # 레이어별 attention map 저장 위치

BASELINE_CKPT = f"{CKPT_DIR}/baseline_best.pth"
GATED_CKPT    = f"{CKPT_DIR}/gated_best.pth"

# ── 분석 ──────────────────────────────────────────────────
TOP_PERCENT  = 10.0    # attention map 상위 N% 강조
NUM_SAMPLES  = 5       # attention 추출에 사용할 검증 이미지 수
