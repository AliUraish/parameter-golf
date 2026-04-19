# Non-record: SP8192 CaseOps + phased TTT reproduction attempt (pre-quant 1.06986890, post-quant SIGSEGV)

This folder preserves a failed but informative reproduction attempt from April 19, 2026.

The run was executed on `8x H100 80GB SXM` and reached a strong diagnostic pre-quantization score, but it crashed with `SIGSEGV` after quantized artifact serialization and before any final scored post-quant evaluation completed.

## Status

- Non-record only
- Not leaderboard-eligible
- No final scored `quantized_*` / `quantized_ttt*` metric was produced

## Recovered Metrics

- In-train stop-point:
  - `4838/20000 val_loss: 2.3423 val_bpb: 1.0703`
- Diagnostic post-EMA pre-quant:
  - `diagnostic pre-quantization post-ema val_loss:2.34142176 val_bpb:1.06986890`
- Artifact size:
  - `Serialized model quantized+brotli: 15911592 bytes`
  - `Total submission size quantized+brotli: 15939907 bytes`

## Failure Mode

The run completed training, EMA application, Hessian collection, and quantized artifact serialization, then crashed before final scored evaluation:

- `Signal 11 (SIGSEGV) received`

Because the final scored evaluation never completed, this result should not be submitted as a record claim.

## Environment

- Hardware: `8x NVIDIA H100 80GB HBM3`
- PyTorch: `2.8.0+cu128`
- CUDA available: `True`

## Included Files

- `README.md` — this note
- `train_seed1234_excerpt.log` — recovered log excerpt from the failed run

## Notes

The exact CaseOps source checkout used on the pod was not preserved in this local workspace. This folder intentionally stores only the verified run evidence available after the failed attempt.
