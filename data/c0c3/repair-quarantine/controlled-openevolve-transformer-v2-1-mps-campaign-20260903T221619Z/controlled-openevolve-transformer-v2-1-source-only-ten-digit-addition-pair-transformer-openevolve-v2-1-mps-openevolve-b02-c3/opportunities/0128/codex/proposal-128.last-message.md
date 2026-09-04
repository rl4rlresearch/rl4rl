MECHANISM: First-head key-basis gauge completion

HYPOTHESIS: A 669-parameter transformer will retain at least 99% accuracy because fixing the third component of one first-head key vector uses the remaining learned Q/K basis freedom while preserving the routing-sensitive second-head query coordinates and all qualified lexical, MLP, and attention-output capacity.

INTENDED_EDIT: Reconstruct the qualified 670-parameter architecture, then fix one additional first-head key-projection coefficient at 0.02.

EVIDENCE: The 670-parameter design achieved 99.78%, while removing another attention-output bias or constraining additional query-bias coordinates failed; the qualified QKV parameterization already fixes the corresponding coefficient in two key channels per head, motivating completion of that gauge in the less-constrained first head.

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
            selected_key_channels.add(2)
        selected_indices = {
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
=======
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

<<<<<<< SEARCH
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=3
        )
=======
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=4
        )
>>>>>>> REPLACE