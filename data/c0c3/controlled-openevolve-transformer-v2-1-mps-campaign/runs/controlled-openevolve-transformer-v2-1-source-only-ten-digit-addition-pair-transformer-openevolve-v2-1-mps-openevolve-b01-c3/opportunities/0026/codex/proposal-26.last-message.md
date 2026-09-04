MECHANISM: Fixed-scale attention with single-neuron MLP width ablation

HYPOTHESIS: Fixing the query-bias amplitude at 0.02 and reducing `d_ff` from 12 to 11 will produce a 1,397-parameter model with at least 99% accuracy, because the verified fixed-bias model achieved 99.96% accuracy and the edit preserves all demonstrated routing dimensions while removing only one MLP neuron.

INTENDED_EDIT: Apply the verified nonlearned 0.02 shared query bias and reduce the default feed-forward width by one.

EVIDENCE: The 1,413-parameter fixed-bias design achieved 99.96% accuracy. Unlike failed reductions to positional or query/key routing capacity, this patch preserves those components and uses that accuracy margin for a conservative 16-parameter MLP ablation.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(n_head, self.head_dim - 1))
=======
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = F.pad(self.q_bias, (0, 1)).reshape(d_model)
=======
        q_bias = F.pad(
            self.q_bias.expand(self.n_head, 1), (0, self.head_dim - 1)
        ).reshape(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE