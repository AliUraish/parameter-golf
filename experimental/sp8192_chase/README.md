# SP8192 Chase

This folder is the low-iteration pivot from the SP1024 starter line to the SP8192 record family.

Files:
- `train_gpt_sp8192_apr05_base.py`
  - Human-readable April 5 SP8192 base script.
  - Good for understanding the stack and doing a first smoke test.
- `train_gpt_sp8192_apr09_record.py`
  - Exact April 9 record script wrapper.
  - This is the direct chase path for the best published result in this family.

Current fallback baseline outside this folder:
- Best 5090 scout on the SP1024 line: `rtx5090_seq2048_fp16embed_leaky992`
- `final_int8_zlib_roundtrip_exact val_bpb = 1.30746963`

## Recommended Path

If credits and time are tight:
1. Skip more SP1024 tuning.
2. Use the exact April 9 record script for the real H100 attempt.
3. Only use the April 5 script for short smoke tests if needed.

## Fresh Pod Setup

Clone and switch to this branch:

```bash
cd /workspace
git clone https://github.com/AliUraish/parameter-golf.git
cd parameter-golf
git checkout codex/sp8192-chase
git pull --ff-only origin codex/sp8192-chase
```

Install the record-matching dependencies:

```bash
cd /workspace/parameter-golf
python3 -m pip install -r requirements.txt
python3 -m pip install brotli sentencepiece
python3 -m pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

Download the SP8192 tokenizer/data from the matched repo:

```bash
cd /workspace/parameter-golf
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

Verify:

```bash
ls -lh data/tokenizers/fineweb_8192_bpe.model
find data/datasets/fineweb10B_sp8192 -name 'fineweb_train_*.bin' | wc -l
find data/datasets/fineweb10B_sp8192 -name 'fineweb_val_*.bin' | wc -l
```

## Short Smoke Test

Use the April 5 base script for a quick one-GPU sanity run:

```bash
cd /workspace/parameter-golf

RUN_ID=sp8192_apr05_smoke \
SEED=1337 \
ITERATIONS=200 \
MAX_WALLCLOCK_SECONDS=120 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 experimental/sp8192_chase/train_gpt_sp8192_apr05_base.py | tee sp8192_apr05_smoke.log
```

## Direct H100 Chase

Run the exact April 9 record script:

```bash
cd /workspace/parameter-golf

SEED=42 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 experimental/sp8192_chase/train_gpt_sp8192_apr09_record.py | tee sp8192_apr09_seed42.log
```

If you want closer reproduction to the record README, keep the script defaults and only override `SEED`, `QK_GAIN_INIT`, and the `TTT_*` knobs.

## What To Save After A Good Run

```bash
grep -E "stopping_early|peak memory allocated|Serialized model|Total submission size|quantized val_loss|quantized_sliding_window|quantized_ttt" logs/*.txt
```

Preserve the exact script and log:

```bash
mkdir -p /workspace/submission_keep/sp8192_chase
cp logs/*.txt /workspace/submission_keep/sp8192_chase/
cp experimental/sp8192_chase/train_gpt_sp8192_apr09_record.py /workspace/submission_keep/sp8192_chase/
cp experimental/sp8192_chase/train_gpt_sp8192_apr05_base.py /workspace/submission_keep/sp8192_chase/
```
