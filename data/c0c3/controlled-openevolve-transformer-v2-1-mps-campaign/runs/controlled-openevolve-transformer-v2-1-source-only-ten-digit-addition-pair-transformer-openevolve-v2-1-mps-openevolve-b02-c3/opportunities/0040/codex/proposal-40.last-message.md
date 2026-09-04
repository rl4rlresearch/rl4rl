MECHANISM: Multi-query causal attention with shared key/value features

HYPOTHESIS: Replacing separate per-head key/value projections with one learned four-dimensional key/value representation will reduce the qualified 1,038-parameter model to 982 parameters while retaining at least 99% accuracy, because head-specific queries, relative biases, attended summaries, and output mixing still provide distinct routing, while digit identity and position can share a common key/value representation.

INTENDED_EDIT: Apply the qualified affine-free pre-MLP normalization and fixed positional amplitude, then replace the 24-output QKV projection with an eight-dimensional query projection and a shared eight-output key/value projection broadcast across both attention heads.

EVIDENCE: The 1,038-parameter design achieved 99.96%, whereas replacing factorized attention with direct quotient-space operators fell to 89.02%; this motivates preserving factorized dot-product attention and head-specific routing while challenging the shared assumption that each head needs independent key and value features. The failed four-coordinate lexical bottleneck also suggests moving compression away from token representation.

<<<<<<< SEARCH
        self.qkv = QuotientInputLinear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
=======
        # Multi-query attention: heads retain distinct learned queries and
        # routing biases while sharing a compact learned key/value vocabulary.
        self.q_proj = QuotientInputLinear(d_model, d_model, bias=False)
        self.kv_proj = QuotientInputLinear(
            d_model, 2 * self.head_dim, bias=False
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q = self.q_proj(x) + self.q_bias
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = QuotientBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_scale = nn.Parameter(torch.tensor(0.02))
=======
        self.register_buffer("pos_scale", torch.tensor(0.02), persistent=False)
>>>>>>> REPLACE