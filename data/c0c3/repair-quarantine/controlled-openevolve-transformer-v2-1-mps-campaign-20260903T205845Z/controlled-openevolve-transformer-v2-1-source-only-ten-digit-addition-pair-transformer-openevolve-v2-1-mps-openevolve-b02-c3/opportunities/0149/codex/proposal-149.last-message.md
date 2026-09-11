MECHANISM: Per-head query-bias gauge fixing

HYPOTHESIS: A 670-parameter transformer will retain at least 99% accuracy because each attention head has its own learned Q/K coordinate basis, and the current 671-parameter design already achieved 99.22% after fixing the equivalent terminal query-bias coordinate in the second head.

INTENDED_EDIT: Reshape the learned query bias by head and fix the final coordinate of each head at zero, removing one parameter while preserving the essential MLP output bias.

EVIDENCE: The current design verified at 99.22% with one query-bias coordinate fixed, whereas removing the remaining MLP output bias caused accuracy to collapse to 19.51%; this motivates an orthogonal, head-symmetric query-bias reduction.

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = QuotientOutputLinear(d_model, d_model)
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(
            torch.zeros(n_head, self.head_dim - 1)
        )
        self.proj = QuotientOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + F.pad(self.q_bias, (0, 1))
=======
        q = q + F.pad(self.q_bias, (0, 1)).reshape(d_model)
>>>>>>> REPLACE