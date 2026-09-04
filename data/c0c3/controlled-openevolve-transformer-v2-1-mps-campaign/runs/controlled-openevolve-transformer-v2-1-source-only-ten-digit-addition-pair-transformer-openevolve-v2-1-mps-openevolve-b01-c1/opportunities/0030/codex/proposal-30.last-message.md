MECHANISM: Hybrid shared/private value representation

HYPOTHESIS: Sharing one learned value channel across the two attention heads, while retaining independent queries, keys, and three private value channels per head, will reduce the verified 1,585-parameter model to 1,577 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Replace one pair of independent value-projection rows with a single deduplicated learned row. Preserve full-width initialization draws and reconstruct the shared row inside causal attention.

EVIDENCE: Fully sharing both key and value heads failed at 36.38%, but that experiment simultaneously removed independent addressing and all private value features. The 99.97%-accurate baseline leaves untested the load-bearing assumption that every value channel must be independent; this patch isolates that assumption while preserving all key capacity and six of eight head-specific value rows.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove the softmax-null key bias
        # and prune one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model : -1])).clone()
        )
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Share one learned value feature across heads while retaining private
        # value features and fully independent query/key projections.
        self.shared_value_dim = 1
        full_weight = self.qkv.weight.detach()
        value_weight = full_weight[2 * d_model :].view(
            n_head, self.head_dim, d_model
        )
        self.qkv.weight = nn.Parameter(
            torch.cat(
                (
                    full_weight[: 2 * d_model],
                    value_weight[0, : self.shared_value_dim],
                    value_weight[:, self.shared_value_dim :].reshape(-1, d_model),
                )
            ).clone()
        )
        self.qkv.shared_value_dim = self.shared_value_dim
        self.qkv.shared_value_n_head = n_head

        # Keep constructor RNG consumption, then remove the softmax-null key bias
        # and prune one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model : -1])).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), F.pad(v_bias, (0, 1)))
        )
        qkv = F.linear(x, self.qkv.weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)
=======
        shared_dim = self.qkv.shared_value_dim
        common_end = 2 * d_model + shared_dim
        qk_weight = self.qkv.weight[: 2 * d_model]
        shared_value_weight = self.qkv.weight[2 * d_model : common_end]
        private_value_weight = self.qkv.weight[common_end:].view(
            self.n_head, self.head_dim - shared_dim, d_model
        )
        value_weight = torch.cat(
            (
                shared_value_weight.unsqueeze(0).expand(
                    self.n_head, -1, -1
                ),
                private_value_weight,
            ),
            dim=1,
        ).reshape(d_model, d_model)
        full_weight = torch.cat((qk_weight, value_weight), dim=0)

        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), F.pad(v_bias, (0, 1)))
        )
        qkv = F.linear(x, full_weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                nn.init.normal_(full, mean=0.0, std=0.02)
                embedding.weight.copy_(full.reshape(-1) @ embedding.basis)
        elif isinstance(module, MeanFreeTokenEmbedding):
=======
                nn.init.normal_(full, mean=0.0, std=0.02)
                embedding.weight.copy_(full.reshape(-1) @ embedding.basis)
        elif isinstance(module, nn.Linear) and hasattr(module, "shared_value_dim"):
            # Draw the original full QKV matrix so initialization RNG ordering
            # is unchanged, then deduplicate one value feature across heads.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                d_model = module.out_features // 3
                n_head = module.shared_value_n_head
                head_dim = d_model // n_head
                shared_dim = module.shared_value_dim
                value = full[2 * d_model :].view(
                    n_head, head_dim, module.in_features
                )
                module.weight.copy_(
                    torch.cat(
                        (
                            full[: 2 * d_model],
                            value[0, :shared_dim],
                            value[:, shared_dim:].reshape(
                                -1, module.in_features
                            ),
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, MeanFreeTokenEmbedding):
>>>>>>> REPLACE