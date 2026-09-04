MECHANISM: Per-head terminal query-coordinate gauge fixing

HYPOTHESIS: A 670-parameter transformer will retain at least 99% accuracy because fixing the terminal query coordinate of the first head mirrors the successful terminal-coordinate fixing in the second head while preserving both remaining query-bias dimensions of the routing-sensitive second head.

INTENDED_EDIT: Replace the five-scalar query bias with four learned scalars and reconstruct a query bias whose final coordinate in each head is fixed at zero.

EVIDENCE: The 671-parameter model reached 99.22% after fixing the second head’s terminal query-bias coordinate, whereas fixing its adjacent coordinate collapsed to 2.33%; this motivates applying the successful coordinate-local constraint symmetrically to the other head instead of further reducing the second head.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = QuotientOutputLinear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(
            torch.zeros(n_head * (self.head_dim - 1))
        )
        self.proj = QuotientOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 1))
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(
            self.q_bias.view(self.n_head, self.head_dim - 1),
            (0, 1),
        ).reshape(d_model)
        q = q + q_bias
>>>>>>> REPLACE