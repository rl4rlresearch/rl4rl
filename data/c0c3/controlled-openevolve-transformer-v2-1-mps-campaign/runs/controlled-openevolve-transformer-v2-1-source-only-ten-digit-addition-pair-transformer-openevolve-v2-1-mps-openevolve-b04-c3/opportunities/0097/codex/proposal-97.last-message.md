MECHANISM: Four-scalar output-side shared-value basis constraint

HYPOTHESIS: Extending the verified 1,138-parameter design by fixing a fourth trailing attention-output projection coefficient at zero will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,137 learned parameters.

INTENDED_EDIT: Reconstruct the final four attention-output projection weights as zeros and use the qualified factorized shared-key/value evaluation to avoid redundant computation while preserving constructor RNG consumption.

EVIDENCE: The factorized three-coefficient design achieved 99.78% accuracy with 1,138 parameters; one additional output-projection constraint is the smallest direct probe of the remaining redundancy.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        # Fix one coefficient on the output side of the shared-value basis.
        # Constructing the full Linear preserves constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 1))
        self.proj.fixed_weight_trim = (0, 1)
=======
        self.proj = nn.Linear(d_model, d_model)
        # Fix four trailing coefficients on the output side of the shared-value
        # basis while preserving the full Linear's constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 4))
        self.proj.fixed_weight_trim = (0, 4)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                query_bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        kv_weight_size = self.head_dim * d_model
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = key_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
        value_weight = F.pad(
            value_weight,
            (0, self.qkv.shared_value_weight_trim),
        ).view(self.head_dim, d_model).repeat(self.n_head, 1)
        qkv_weight = torch.cat((query_weight, key_weight, value_weight), dim=0)
        qkv = F.linear(x, qkv_weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        kv_weight_size = self.head_dim * d_model
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = key_weight.view(self.head_dim, d_model)
        value_weight = F.pad(
            value_weight,
            (0, self.qkv.shared_value_weight_trim),
        ).view(self.head_dim, d_model)
        q = F.linear(x, query_weight, query_bias)
        shared_kv = F.linear(
            x,
            torch.cat((key_weight, value_weight), dim=0),
        )
        k, v = shared_kv.chunk(2, dim=-1)
        k = k.repeat(1, 1, self.n_head)
        v = v.repeat(1, 1, self.n_head)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_weight = F.pad(self.proj.weight, (0, 1)).view(d_model, d_model)
=======
        proj_weight = F.pad(self.proj.weight, (0, 4)).view(d_model, d_model)
>>>>>>> REPLACE