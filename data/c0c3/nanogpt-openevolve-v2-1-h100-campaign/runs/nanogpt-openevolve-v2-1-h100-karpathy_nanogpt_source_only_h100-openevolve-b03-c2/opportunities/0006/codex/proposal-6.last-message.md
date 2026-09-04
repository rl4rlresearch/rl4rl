MECHANISM: Token-normalized Muon momentum ramp

HYPOTHESIS: Restoring the efficient 128-sequence, 262K-token update and extending the Muon momentum ramp from 300 to 600 steps will beat val_bpb 0.98713 by preserving the best throughput while keeping the momentum transition at approximately the baseline’s token horizon.

INTENDED_EDIT: Restore Reference Design 3’s single-microbatch configuration and double the momentum-ramp step count to compensate for its roughly doubled optimizer-update frequency.

EVIDENCE: Reference Design 3 achieved the best val_bpb, 0.98713, with 1,868 steps and 489.7M tokens; its unchanged 300-step momentum ramp reaches 0.95 after half as many training tokens as the 524K-token baseline, while the 96- and 192-sequence designs show that changing away from the efficient 128-sequence microbatch sharply reduces throughput.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**17 # ~393K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 192  # one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
=======
def get_muon_momentum(step):
    frac = min(step / 600, 1)
    return (1 - frac) * 0.85 + frac * 0.95
>>>>>>> REPLACE