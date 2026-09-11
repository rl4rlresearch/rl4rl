MECHANISM: Interpolated MLP capacity–throughput balance

HYPOTHESIS: A 4.375× MLP will process at least 465M tokens and reduce val_bpb below 0.994680 by recovering throughput while retaining most of the capacity benefit of the 4.5× design.

INTENDED_EDIT: Narrow each 512-dimensional MLP from 2304 to a Tensor-Core-aligned 2240 hidden channels.

EVIDENCE: Moving from 4.25× to 4.5× improved val_bpb by only 0.000073 while losing 11.1M tokens, indicating diminishing returns and placing the likely optimum between those widths.

<<<<<<< SEARCH
        hidden_dim = 9 * config.n_embd // 2
=======
        hidden_dim = 35 * config.n_embd // 8
>>>>>>> REPLACE