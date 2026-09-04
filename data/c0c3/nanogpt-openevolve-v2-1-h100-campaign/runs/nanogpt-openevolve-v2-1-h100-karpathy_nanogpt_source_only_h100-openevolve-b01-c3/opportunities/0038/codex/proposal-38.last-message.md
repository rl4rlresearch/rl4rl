MECHANISM: Below-optimum MLP width interpolation

HYPOTHESIS: A 4.875× MLP will process at least 455M tokens and reduce val_bpb below 0.982905 by preserving most of the 5× model’s capacity while increasing token exposure.

INTENDED_EDIT: Reduce every MLP hidden width from 5.25× to 4.875× (2496 channels), retaining the proven architecture, batch, schedule, and 93.75% learning rates.

EVIDENCE: The 5× MLP achieved the best val_bpb of 0.982905; 5.125×, 5.25×, and 5.625× all regressed, while 4.375× was also worse. Testing 4.875× completes the nearest symmetric bracket around the observed 5× optimum.

<<<<<<< SEARCH
        hidden_dim = 21 * config.n_embd // 4
=======
        hidden_dim = 39 * config.n_embd // 8
>>>>>>> REPLACE