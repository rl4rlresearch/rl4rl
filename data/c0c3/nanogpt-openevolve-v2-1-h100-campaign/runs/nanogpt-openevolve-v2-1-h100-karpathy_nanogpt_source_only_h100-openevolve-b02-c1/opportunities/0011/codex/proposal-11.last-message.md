MECHANISM: Channel-wise lexical-residual readout

HYPOTHESIS: Expanding the successful direct lexical expert from four head-level coefficients to 512 channel-level coefficients will improve its ability to select predictive token features and reduce val_bpb below 0.995511 without materially affecting throughput.

INTENDED_EDIT: Replace the final value embedding’s per-head readout scaling with independent scaling for every model channel, retaining the same initialization, architecture, and optimizer.

EVIDENCE: The per-head lexical-residual readout improved val_bpb from 0.995558 to 0.995511 at similar throughput, while the faster grouped-query design regressed to 0.999976; this motivates increasing value-path expressivity with negligible added computation.

<<<<<<< SEARCH
        # Per-head strength of the direct lexical (bigram) prediction expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_head,), 0.1))
=======
        # Per-channel strength of the direct lexical (bigram) prediction expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_embd,), 0.1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The final layer always has a value embedding. Route it directly to the
        # readout as a per-head low-rank bigram expert alongside contextual state.
        lexical = norm(ve).view(B, T, self.config.n_head, -1)
        lexical = lexical * self.bigram_lambdas.view(1, 1, -1, 1)
        x = norm(x + lexical.reshape(B, T, -1))
=======
        # The final layer always has a value embedding. Route it directly to the
        # readout with channel-wise scaling alongside the contextual state.
        lexical = norm(ve)
        lexical = lexical * self.bigram_lambdas.view(1, 1, -1)
        x = norm(x + lexical)
>>>>>>> REPLACE