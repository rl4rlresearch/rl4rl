MECHANISM: Three-times-width ReLU² feed-forward compute rebalancing

HYPOTHESIS: Narrowing each MLP from 4× to 3× while restoring the proven 50% linear warmdown will preserve the established attention geometry, exceed 513.3M training tokens without reducing optimizer-step frequency, and lower val_bpb below 0.993287.

INTENDED_EDIT: Reduce MLP hidden width from 4× to 3× model dimension and restore WARMDOWN_RATIO from 0.6 to the best verified value of 0.5.

EVIDENCE: The 512-token SSSL design with 50% linear warmdown achieved the best val_bpb, 0.993287 at 513.3M tokens; doubling the batch reached 519.0M but regressed to 1.015936 because it halved update frequency, motivating throughput gains from cheaper dense layers instead.

<<<<<<< SEARCH
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
=======
        self.c_fc = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(3 * config.n_embd, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
>>>>>>> REPLACE