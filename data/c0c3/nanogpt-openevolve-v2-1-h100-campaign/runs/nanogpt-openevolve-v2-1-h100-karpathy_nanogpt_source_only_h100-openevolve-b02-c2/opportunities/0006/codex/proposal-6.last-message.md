MECHANISM: Token-horizon-preserving Muon momentum ramp

HYPOTHESIS: Restoring the proven 1024-token local windows and 262K-token batch while extending Muon’s momentum ramp from 300 to 600 updates will lower val_bpb below 0.987182 by preserving the original 524K-batch momentum trajectory per token.

INTENDED_EDIT: Restore the best-performing half-context attention and single 128-sequence microbatch, then double the momentum-ramp steps to account for the doubled optimizer-update cadence.

EVIDENCE: The 262K-token design achieved the best val_bpb of 0.987182 at nearly unchanged throughput, whereas 512-token windows and smaller device batches worsened results; its halved batch doubles updates per token while the existing momentum schedule remains step-based.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 2
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step; one device batch
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
=======
def get_muon_momentum(step):
    # Preserve the 524K-batch momentum trajectory in tokens after halving the batch.
    frac = min(step / 600, 1)
    return (1 - frac) * 0.85 + frac * 0.95
>>>>>>> REPLACE