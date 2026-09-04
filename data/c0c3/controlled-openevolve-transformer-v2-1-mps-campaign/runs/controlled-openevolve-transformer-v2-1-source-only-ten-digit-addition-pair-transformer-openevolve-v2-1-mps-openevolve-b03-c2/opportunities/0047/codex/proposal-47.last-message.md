MECHANISM: Single-head concentrated query-basis gauge fixing

HYPOTHESIS: Concentrating both fixed query-bias coordinates in the already-gauged second head will produce a 1504-parameter model with at least 99% accuracy, because it preserves a fully unconstrained first head while using the second head’s within-head query/key basis symmetry.

INTENDED_EDIT: Apply the verified final-LayerNorm scale gauge and store six query-bias coordinates, reconstructing the final two coordinates of the second head as zero.

EVIDENCE: The single-coordinate query gauge reached 99.97% at 1505 parameters, while distributing two fixed coordinates across both heads reached 98.61% at 1504; concentrating both constraints in one head tests whether leaving one head fully unconstrained restores the missing margin.

<<<<<<< SEARCH
        # Key bias is softmax-invariant, and every value-bias coordinate can
        # be absorbed by the downstream projection bias. Store only query
        # bias and reconstruct the other two bias vectors in fixed gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        # Key bias is softmax-invariant, value bias is absorbable, and an
        # invertible query/key basis change within the second head can fix
        # two of its query-bias coordinates while leaving the first head
        # completely unconstrained.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = self.qkv.bias[:d_model]
        value_bias = self.qkv.bias.new_zeros(d_model)
=======
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2))
        )
        value_bias = self.qkv.bias.new_zeros(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
=======
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        # Fix one common positive affine scale, which changes only global
        # logit temperature under protected argmax decoding.
        self.ln_f = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 1))
        self.ln_f_bias = nn.Parameter(torch.zeros(cfg.d_model))

        # Weight tying with input embeddings.
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = F.linear(x, token_weight)
=======
        x = self.ln_f(x)
        ln_f_weight = torch.cat(
            (
                self.ln_f_weight,
                self.ln_f_weight.new_ones(1),
            )
        )
        x = x * ln_f_weight + self.ln_f_bias
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE