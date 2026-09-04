MECHANISM: Seven-coordinate residual stream with preserved rank-five vocabulary code

HYPOTHESIS: Reducing `d_model` from eight to seven while preserving the verified rank-five embedding, six-unit MLP, and four learned secondary-head gains will retain at least 99% accuracy and reduce parameters from 747 to 721.

INTENDED_EDIT: Remove one ambient residual coordinate, retain five learned vocabulary coordinates, resize learned projections naturally, and adapt attention gain/bias expansions by removing one repeated coordinate.

EVIDENCE: The 747-parameter model reached 99.89%, while reducing the MLP to five units collapsed to 5.6%; this suggests nonlinear width is load-bearing but does not establish that two residual coordinates beyond the rank-five vocabulary subspace are necessary. The scalar-address/shared-value attention also does not actually require `d_model` to divide the head count.

<<<<<<< SEARCH
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, embedding_dim - 3)
        self.register_buffer(
            "vocab_basis", mean_zero_basis(num_embeddings), persistent=False
        )
        self.register_buffer(
            "feature_basis",
            mean_zero_basis(embedding_dim)[:, :-2],
            persistent=False,
        )
=======
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, 5)
        self.register_buffer(
            "vocab_basis", mean_zero_basis(num_embeddings), persistent=False
        )
        self.register_buffer(
            "feature_basis",
            mean_zero_basis(embedding_dim)[:, :5],
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if n_head < 2:
            raise ValueError("n_head must be at least two")

        self.n_head = n_head
        self.query_dim = 1
        self.output_dim = d_model - 1
        self.address_scale = nn.Parameter(torch.full((1,), 0.02))
        self.value = MeanZeroInputLinear(d_model, self.output_dim)
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 3)
        )
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        if d_model != 7:
            raise ValueError("d_model must be seven")
        if n_head < 2:
            raise ValueError("n_head must be at least two")

        self.n_head = n_head
        self.query_dim = 1
        self.output_dim = d_model - 1
        self.address_scale = nn.Parameter(torch.full((1,), 0.02))
        self.value = MeanZeroInputLinear(d_model, self.output_dim)
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 2)
        )
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(self.output_dim - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :1],
                self.secondary_value_gain[..., 1:].repeat_interleave(2, dim=-1),
            ),
            dim=-1,
        )
=======
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :1],
                self.secondary_value_gain[..., 1:3].repeat_interleave(2, dim=-1),
                self.secondary_value_gain[..., -1:],
            ),
            dim=-1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (
                self.output_bias[:1],
                self.output_bias[1:2].expand(2),
                self.output_bias[-1:].expand(4),
            )
        )
=======
        bias = torch.cat(
            (
                self.output_bias[:1],
                self.output_bias[1:2].expand(2),
                self.output_bias[-1:].expand(3),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-model", type=int, default=8)
=======
    p.add_argument("--d-model", type=int, default=7)
>>>>>>> REPLACE