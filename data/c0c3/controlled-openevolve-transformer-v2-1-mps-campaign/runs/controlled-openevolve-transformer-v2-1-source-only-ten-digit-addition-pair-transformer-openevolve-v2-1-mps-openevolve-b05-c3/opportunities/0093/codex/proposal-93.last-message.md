MECHANISM: Fixed-spacing phase-shared lag attention

HYPOTHESIS: Fixing the relative head phase at its evenly spaced initialization will retain at least 99% accuracy while reducing the model from 1,257 to 1,256 learned parameters, because the unrestricted learned lag kernel can adapt its routing while the fixed nonzero phase preserves head specialization.

INTENDED_EDIT: Replace the learned relative-shift parameter with a fixed buffer, leaving the learned lag kernel, causal attention, optimizer, checkpoints, and decoding interface unchanged.

EVIDENCE: Equal-temperature phase-shared attention achieved 99.93% accuracy with 1,257 parameters, showing that distinct cyclic phase routing remains sufficient after removing head-specific temperature; fixing only the remaining scalar phase is the most direct next boundary test.

<<<<<<< SEARCH
        self.relative_shift = nn.Parameter(
            torch.arange(1, n_head, dtype=torch.float32)
            * (max_seq_len / n_head)
        )
=======
        self.register_buffer(
            "relative_shift",
            torch.arange(1, n_head, dtype=torch.float32)
            * (max_seq_len / n_head),
        )
>>>>>>> REPLACE