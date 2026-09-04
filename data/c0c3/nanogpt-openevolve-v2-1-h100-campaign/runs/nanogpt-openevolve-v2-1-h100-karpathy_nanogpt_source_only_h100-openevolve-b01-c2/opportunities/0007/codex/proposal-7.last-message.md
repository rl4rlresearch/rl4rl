MECHANISM: Alternating one- and two-microbatch optimizer steps

HYPOTHESIS: Alternating 262K- and 524K-token updates will combine the current design’s higher update frequency with Reference Design 1’s higher throughput, processing roughly 460M tokens in about 1,200 steps and lowering val_bpb below 0.995558.

INTENDED_EDIT: Keep the proven SSSL architecture and 128-sequence microbatch, but alternate gradient accumulation between one and two microbatches while accurately tracking tokens and MFU.

EVIDENCE: One-microbatch training reached 0.996687 with 1,560 steps and 408.9M tokens, while two-microbatch training reached 0.995558 with 948 steps and 497.0M tokens; an alternating batch schedule directly tests the unobserved throughput/update-frequency midpoint.

<<<<<<< SEARCH
# Optimization
TOTAL_BATCH_SIZE = 2**18 # 262K tokens; one 128-sequence microbatch per step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
=======
# Optimization
GRAD_ACCUM_PATTERN = (1, 2) # alternate 262K- and 524K-token optimizer steps
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
>>>>>>> REPLACE

<<<<<<< SEARCH
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
=======
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert len(GRAD_ACCUM_PATTERN) > 0 and all(n >= 1 for n in GRAD_ACCUM_PATTERN)
>>>>>>> REPLACE

<<<<<<< SEARCH
print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
=======
print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation pattern: {GRAD_ACCUM_PATTERN}")
print(f"Average optimizer batch: {tokens_per_fwdbwd * sum(GRAD_ACCUM_PATTERN) / len(GRAD_ACCUM_PATTERN):,.0f} tokens")
>>>>>>> REPLACE

<<<<<<< SEARCH
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
=======
total_training_time = 0
total_tokens = 0
measured_tokens = 0
step = 0

while True:
    grad_accum_steps = GRAD_ACCUM_PATTERN[step % len(GRAD_ACCUM_PATTERN)]
    tokens_this_step = tokens_per_fwdbwd * grad_accum_steps
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step > 10:
        total_training_time += dt
=======
    total_tokens += tokens_this_step
    if step > 10:
        total_training_time += dt
        measured_tokens += tokens_this_step
>>>>>>> REPLACE

<<<<<<< SEARCH
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
=======
    tok_per_sec = int(tokens_this_step / dt)
    mfu = 100 * num_flops_per_token * tokens_this_step / dt / H100_BF16_PEAK_FLOPS
>>>>>>> REPLACE

<<<<<<< SEARCH
print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
=======
print()  # newline after \r training log

# Final eval
>>>>>>> REPLACE

<<<<<<< SEARCH
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
=======
steady_state_mfu = 100 * num_flops_per_token * measured_tokens / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
>>>>>>> REPLACE