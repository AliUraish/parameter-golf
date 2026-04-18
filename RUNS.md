# Unified Runbook

This branch contains:

- the root `train_gpt.py` / `train_gpt_mlx.py` from `codex/root-starter-improvements`
- the SP8192 scout scripts in `experimental/sp8192_chase/`

Current reference score to beat on `5090`:

- `rtx5090_seq2048_fp16embed_leaky992`
- `final_int8_zlib_roundtrip_exact val_bpb = 1.30746963`

## Pod Update

```bash
cd /workspace/parameter-golf
git fetch origin
git checkout codex/unified-setups
git pull --ff-only origin codex/unified-setups
mkdir -p logs
```

## Best Root 5090 Baseline

```bash
cd /workspace/parameter-golf

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RUN_ID=rtx5090_seq2048_fp16embed_leaky992 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
MLP_MULT=2 \
MLP_HIDDEN=992 \
MLP_LEAKY_SLOPE=0.5 \
NUM_KV_HEADS=4 \
QK_GAIN_INIT=1.5 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=196608 \
ITERATIONS=4000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=196608 \
TRAIN_LOG_EVERY=100 \
WARMUP_STEPS=20 \
WARMDOWN_ITERS=3600 \
EMA_BETA=0.0 \
TIED_EMBED_LR=0.04 \
MATRIX_LR=0.055 \
SCALAR_LR=0.032 \
FP16_EMBED_EXPORT=1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee rtx5090_seq2048_fp16embed_leaky992.log
```

Extract:

```bash
grep -E "step:[0-9]+/[0-9]+ val_loss:|stopping_early|peak memory allocated|Serialized model int8\\+zlib|Total submission size int8\\+zlib|final_int8_zlib_roundtrip_exact|final_weights:" logs/rtx5090_seq2048_fp16embed_leaky992.txt
```

## SP8192 Data Setup

```bash
cd /workspace/parameter-golf
python3 -m pip install -r requirements.txt
python3 -m pip install --no-cache-dir flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280
python3 -m pip install brotli sentencepiece
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

## Apr08 5090 Gate

Use this only as a cheap gate. It disables the slow sliding-window and TTT tail.

```bash
cd /workspace/parameter-golf

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RUN_ID=sp8192_apr08_5090_gate \
SEED=42 \
PARALLEL_START_LAYER=7 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=98304 \
VAL_BATCH_TOKENS=65536 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
SLIDING_WINDOW_ENABLED=0 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 experimental/sp8192_chase/train_gpt_sp8192_apr08_direct.py | tee sp8192_apr08_5090_gate.log
```

Extract:

```bash
grep -E "use_torch_compile:|stopping_early|peak memory allocated|Serialized model|Total submission size|pre-quantization post-ema|quantized val_loss|val_bpb" logs/sp8192_apr08_5090_gate.txt
```

## Apr09 5090 Gate

Use this only as a cheap gate. It disables the slow sliding-window and TTT tail.

```bash
cd /workspace/parameter-golf

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RUN_ID=sp8192_apr09_5090_gate \
SEED=42 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=98304 \
VAL_BATCH_TOKENS=65536 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
QK_GAIN_INIT=5.25 \
SLIDING_WINDOW_ENABLED=0 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 experimental/sp8192_chase/train_gpt_sp8192_apr09_direct.py | tee sp8192_apr09_5090_gate.log
```

Extract:

```bash
grep -E "use_torch_compile:|stopping_early|peak memory allocated|Serialized model|Total submission size|pre-quantization post-ema|quantized val_loss|val_bpb" logs/sp8192_apr09_5090_gate.txt
```

## Promotion Rule

- If a gate run does not beat `1.30746963`, stop.
- If a gate run beats `1.30746963`, rerun that same script with:
  - `SLIDING_WINDOW_ENABLED=1`
  - `TTT_ENABLED=1`
- Use `H100` for real leaderboard judgement on SP8192. The `5090` is only for smoke and gating on this family.
