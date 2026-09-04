MECHANISM: LayerNorm-null query projection with absorbed attention input bias

HYPOTHESIS: Removing `ln1_bias`, restoring all eight direct query-bias coordinates, and centering every query-weight row will reduce learned parameters from 1,453 to 1,439 while retaining at least 99% accuracy, because the attention input then comes directly from zero-mean LayerNorm, while the removed constant query and value effects remain representable by the existing direct query and centered output biases.

INTENDED_EDIT: Delete the redundant eight-parameter pre-attention bias, expand the query bias from six to eight parameters, and store query weights in the same seven-dimensional zero-mean basis already used successfully for key and value weights.

EVIDENCE: Zero-mean input parameterizations retained 100% accuracy for key weights and 99.95% for MLP input weights, while the current direct centered attention output bias reached 99.97%; together these results support applying the same LayerNorm-nullspace constraint to queries after absorbing the redundant pre-attention affine bias.

<<<<<<< SEARCH
        # Construct the original affines first to preserve the baseline RNG
        # stream, then retain six query-bias coordinates. Under zero attention
        # dropout, a value bias passes through aggregation unchanged, so store
        # its observable effect directly as a centered output bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(d_model - 2))
=======
        # Construct the original affine first to preserve the baseline RNG
        # stream. The Block supplies normalized inputs directly, so retain a
        # full direct query bias while storing all three input projections in
        # the observable zero-mean LayerNorm subspace.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
            q_weight = weight[:d_model]
            k_weight = weight[d_model : 2 * d_model]
            v_weight = weight[2 * d_model :]
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_weight.reshape(-1),
                        k_coeff.reshape(-1),
                        v_coeff.reshape(-1),
                    )
                ).clone()
            )
=======
            q_weight = weight[:d_model]
            k_weight = weight[d_model : 2 * d_model]
            v_weight = weight[2 * d_model :]
            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = q_weight @ self.proj_basis
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_coeff.reshape(-1),
                        k_coeff.reshape(-1),
                        v_coeff.reshape(-1),
                    )
                ).clone()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_size = d_model * d_model
        k_size = d_model * (d_model - 1)
        q_weight = self.qkv.weight[:q_size].view(d_model, d_model)
        k_coeff = self.qkv.weight[q_size : q_size + k_size].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_coeff = self.qkv.weight[q_size + k_size :].view(
            d_model, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
        q = F.linear(x, q_weight)
        k = F.linear(x, k_weight)
        v = F.linear(x, v_weight)
        q_bias = torch.cat((self.qkv.bias, self.qkv.bias.new_zeros(2)))
        q = q + q_bias
=======
        matrix_size = d_model * (d_model - 1)
        q_coeff = self.qkv.weight[:matrix_size].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T
        k_coeff = self.qkv.weight[matrix_size : 2 * matrix_size].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_coeff = self.qkv.weight[2 * matrix_size :].view(
            d_model, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
        q = F.linear(x, q_weight, self.qkv.bias)
        k = F.linear(x, k_weight)
        v = F.linear(x, v_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.ln1_bias = nn.Parameter(torch.zeros(cfg.d_model))
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = x + self.attn(self.ln1(x) + self.ln1_bias)
=======
        x = x + self.attn(self.ln1(x))
>>>>>>> REPLACE