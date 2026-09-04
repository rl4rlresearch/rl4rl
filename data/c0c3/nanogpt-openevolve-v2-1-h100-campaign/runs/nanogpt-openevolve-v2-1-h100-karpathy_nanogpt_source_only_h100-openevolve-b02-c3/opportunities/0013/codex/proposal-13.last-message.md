MECHANISM: Large-batch optimizer-step amortization

HYPOTHESIS: Doubling the optimizer batch to 1.05M tokens will amortize Muon overhead, process more than 513.3M tokens in five minutes, and reduce val_bpb below 0.993287 despite fewer parameter updates.

INTENDED_EDIT: Double TOTAL_BATCH_SIZE while preserving the proven 512-token SSSL architecture, learning rates, and 50% linear warmdown.

EVIDENCE: Halving the batch increased updates from 979 to 1463 but reduced training volume from 513.3M to 383.5M tokens and regressed val_bpb from 0.993287 to 1.000127, indicating that optimizer overhead and token throughput outweigh additional update frequency.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**20 # ~1.05M tokens per optimizer step
>>>>>>> REPLACE