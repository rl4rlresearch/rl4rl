MECHANISM: Depth-preserving feed-forward contraction

HYPOTHESIS: Reducing each MLP from 4× to 3.5× width while retaining all eight attention blocks will increase token throughput enough to achieve `val_bpb < 0.986636`.

INTENDED_EDIT: Change the MLP hidden dimension from 2048 to 1792 at the current 512-wide model, preserving tensor-core alignment and every other setting.

EVIDENCE: Removing an entire block increased throughput to 539.9M tokens but regressed to `0.990593`, while adding a block collapsed throughput to 369.6M tokens; a modest MLP contraction tests a compute saving that does not sacrifice depth.

<<<<<<< SEARCH
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
=======
        hidden_dim = 7 * config.n_embd // 2
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
>>>>>>> REPLACE