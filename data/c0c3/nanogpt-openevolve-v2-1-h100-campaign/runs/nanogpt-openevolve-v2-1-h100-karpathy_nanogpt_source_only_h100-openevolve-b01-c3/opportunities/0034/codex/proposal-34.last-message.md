MECHANISM: Tensor-core-aligned MLP width interpolation

HYPOTHESIS: A 5.25× MLP will retain at least 440M-token throughput and reduce val_bpb below 0.982905.

INTENDED_EDIT: Increase every MLP hidden width from 4.375× to 5.25× while retaining the proven 262,144-token batch, eight-layer architecture, linear warmdown, and 93.75% learning rates.

EVIDENCE: Increasing MLP width from 4.375× to 5× improved val_bpb from 0.984614 to 0.982905 despite lower token exposure, while 5.625× regressed to 0.983497 after throughput fell to 424.1M tokens; the 5.25× width is a conservative interpolation and remains aligned to 128-channel tensor-core tiles.

<<<<<<< SEARCH
        hidden_dim = 35 * config.n_embd // 8
=======
        hidden_dim = 21 * config.n_embd // 4
>>>>>>> REPLACE