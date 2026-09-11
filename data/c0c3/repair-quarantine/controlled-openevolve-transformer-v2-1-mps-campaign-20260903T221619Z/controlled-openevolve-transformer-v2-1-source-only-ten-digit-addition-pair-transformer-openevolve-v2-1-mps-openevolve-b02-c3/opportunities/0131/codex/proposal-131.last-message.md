MECHANISM: Second-head key-basis gauge completion

HYPOTHESIS: A 668-parameter transformer will retain at least 99% accuracy because fixing the third first-input key coefficient in the second head mirrors the successful first-head gauge completion while preserving independent head-specific keys, queries, values, and positional routing.

INTENDED_EDIT: Reconstruct the qualified 669-parameter design, then fix the corresponding third key-channel coefficient in both heads, removing one additional learned scalar from the qualified model.

EVIDENCE: The 669-parameter first-head key-basis design achieved 99.70%; unlike shared-key compression, which fell to 92.50%, this extends the successful within-head Q/K gauge constraint without coupling the heads or removing routing-sensitive query coordinates.

<<<<<<< SEARCH
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
        selected_indices = {
=======
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
        if head_dim >= 3:
            selected_key_channels.update(
                head * head_dim + 2 for head in range(n_head)
            )
        selected_indices = {
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = QuotientOutputLinear(
            d_model, d_model, fixed_bias_coeffs=1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 1))
>>>>>>> REPLACE