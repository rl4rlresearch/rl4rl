MECHANISM: Fixed-direction scalar content addressing

HYPOTHESIS: A 1,148-parameter transformer will achieve at least 99% accuracy because the verified 1,155-parameter model reached 100% with one tied scalar query-key feature, and the learned embeddings can align that feature with a fixed input coordinate while a learned gain preserves adaptive attention strength.

INTENDED_EDIT: Replace the eight-parameter tied scalar address projection with a one-parameter gain applied to the first normalized feature coordinate, while preserving shared query-key addressing, full-width values, and eleven-bin positional routing.

EVIDENCE: The 1,155-parameter tied scalar query-key design achieved 100% accuracy after successive reductions of addressing width and head-specific projections all succeeded, motivating removal of the remaining learned address-direction degrees without altering the proven positional, value, MLP, normalization, or residual components.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.query_dim = max(1, self.head_dim // 2)
        self.qk = nn.Linear(d_model, 2 * n_head * self.query_dim, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.query_dim = 1
        self.address_scale = nn.Parameter(torch.full((1,), 0.02))
        self.value = nn.Linear(d_model, d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        q, k = self.qk(x).chunk(2, dim=-1)
        v = self.value(x)
        q_bias = F.pad(
            self.q_bias.expand(self.n_head, 1), (0, self.query_dim - 1)
        ).reshape(self.n_head * self.query_dim)
        q = q + q_bias

        q = q.view(bsz, seqlen, self.n_head, self.query_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.query_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        bsz, seqlen, d_model = x.shape
        address = self.address_scale * x[..., :1]
        v = self.value(x)

        q = address.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        k = address.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        q = q + self.q_bias.view(1, 1, 1, 1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE