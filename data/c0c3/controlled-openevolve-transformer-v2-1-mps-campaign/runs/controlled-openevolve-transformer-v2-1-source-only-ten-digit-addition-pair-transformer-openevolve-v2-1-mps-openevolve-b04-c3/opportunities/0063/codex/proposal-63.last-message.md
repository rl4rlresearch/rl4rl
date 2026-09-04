MECHANISM: Single-entry query/key basis gauge fixing

HYPOTHESIS: Fixing one query-weight coordinate at zero on the qualified 1,346-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,345 parameters.

INTENDED_EDIT: Adopt the qualified three-parameter shared query bias and seven-parameter projection bias, then remove one query-weight scalar and reconstruct the full QKV matrix with a fixed zero while preserving initialization RNG consumption.

EVIDENCE: Reference Design 3 achieved 100% accuracy with 1,346 parameters. Since reducing the shared query bias further failed at 74.7%, this instead tests an independent Q/K basis-change redundancy while leaving the successful token, position, and optimization backbone intact.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Independent Q/K basis changes permit three
        # corresponding query-bias coordinates to be shared across the heads.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 3))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key and value biases are redundant. Q/K basis changes permit one
        # query-weight entry to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 1))
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # The two token-free residual channels provide one rotational gauge.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = torch.cat(
            (
                self.qkv.bias[: self.head_dim],
                self.qkv.bias[:3],
                self.qkv.bias[self.head_dim :],
            )
        )
        qkv_bias = torch.cat(
            (
                query_bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
        qkv_bias = torch.cat(
            (
                query_bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
        qkv_weight = F.pad(self.qkv.weight, (1, 0)).view(3 * d_model, d_model)
        qkv = F.linear(x, qkv_weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_bias = F.pad(self.proj.bias, (0, 1))
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.out_features * module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[1:])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE