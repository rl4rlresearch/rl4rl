MECHANISM: Tied scalar query-key content addressing

HYPOTHESIS: A 1,155-parameter transformer will achieve at least 99% accuracy because the verified 1,163-parameter model reached 99.99% with head-shared scalar queries and keys, leaving one scalar content feature that can serve both roles while positional routing and full-width values remain unchanged.

INTENDED_EDIT: Replace the separate head-specific scalar query/key projection with one eight-parameter projection shared across both heads and both addressing roles.

EVIDENCE: The 1,163-parameter design achieved 99.99% after independently sharing query and key projections across heads; tying those two remaining scalar projections is the next isolated reduction and leaves the proven eleven-bin relative biases, full-width values, MLP, normalization, and residual projections intact.

<<<<<<< SEARCH
        self.query_dim = 1
        self.qk = nn.Linear(d_model, 2 * n_head * self.query_dim, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
=======
        self.query_dim = 1
        self.address = nn.Linear(d_model, self.query_dim, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k = self.qk(x).chunk(2, dim=-1)
        v = self.value(x)
        q_bias = F.pad(
            self.q_bias.expand(self.n_head, 1), (0, self.query_dim - 1)
        ).reshape(self.n_head * self.query_dim)
        q = q + q_bias

        q = q.view(bsz, seqlen, self.n_head, self.query_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.query_dim).transpose(1, 2)
=======
        address = self.address(x)
        v = self.value(x)

        q = address.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        k = address.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        q = q + self.q_bias.view(1, 1, 1, 1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
>>>>>>> REPLACE