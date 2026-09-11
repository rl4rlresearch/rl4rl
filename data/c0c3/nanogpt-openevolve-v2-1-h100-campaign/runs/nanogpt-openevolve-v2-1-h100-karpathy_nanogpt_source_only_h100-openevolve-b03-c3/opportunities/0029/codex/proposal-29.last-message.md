MECHANISM: Decayed causal-context warm start

HYPOTHESIS: Initializing the proven two-lag FIR with small decaying coefficients will lower val_bpb below 0.992110 while retaining at least 470M tokens by exposing useful local context immediately instead of learning both lag paths from zero.

INTENDED_EDIT: Preserve the best four handoff gates and two-lag architecture, but initialize the one-token and two-token embedding coefficients to 0.10 and 0.05 respectively.

EVIDENCE: Fixed two-lag mixing contributed to the best 0.992110 result, whereas a third lag regressed to 0.992455 and token-conditioned scaling reached only 0.992360; this motivates accelerating the established fixed two-lag path without adding computation or parameters.

<<<<<<< SEARCH
        # Residual and causal embedding-mixing parameters
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        self.embedding_context.zero_()
=======
        # Warm-start the proven local-context path with decaying causal weights.
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        self.embedding_context[0].fill_(0.10)
        self.embedding_context[1].fill_(0.05)
>>>>>>> REPLACE