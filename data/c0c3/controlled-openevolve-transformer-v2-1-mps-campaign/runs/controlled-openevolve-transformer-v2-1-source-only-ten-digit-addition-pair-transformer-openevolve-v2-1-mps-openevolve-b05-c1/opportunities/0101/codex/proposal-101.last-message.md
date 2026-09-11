MECHANISM: Runtime-streamlined third orthonormal attention-null reuse

HYPOTHESIS: Reusing projection column 3’s normalized row-common component as the remaining final-LayerNorm scale will produce a 1,265-parameter model with at least 99% accuracy; eliminating zero-dropout calls and duplicate token-weight reconstruction will allow the previously timed-out exact reparameterization to finish verification.

INTENDED_EDIT: Remove the dedicated final-LayerNorm scale, derive it from projection column 3, batch the normalized column reductions, and streamline function-preserving operations for the fixed zero-dropout configuration.

EVIDENCE: Orthonormal attention-null reuse achieved 100% accuracy at both 1,267 and 1,266 parameters. The same third reuse timed out twice without contrary accuracy evidence, motivating preservation of that exact symmetry while reducing hot-path overhead.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with scales and bias stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))

    def forward(
        self,
        x: torch.Tensor,
        shared_scales: torch.Tensor,
        shared_bias: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                shared_scales[1:2],
                self.weight.new_ones(2),
                shared_scales[:1],
                self.weight.new_ones(2),
            )
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with scales and bias stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)

    def forward(
        self,
        x: torch.Tensor,
        shared_scales: torch.Tensor,
        shared_bias: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                shared_scales[2:3],
                shared_scales.new_ones(1),
                shared_scales[1:2],
                shared_scales.new_ones(2),
                shared_scales[:1],
                shared_scales.new_ones(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
=======
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y
=======
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        return self.proj(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = MeanZeroInputLinear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, shared_biases: torch.Tensor) -> torch.Tensor:
        output_bias = torch.cat((self.output_bias, shared_biases.reshape(-1)))
        hidden = self.fc1(x) + output_bias.mean()
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        output = F.pad(output, (0, 1))
        return self.drop(output)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))

    def forward(self, x: torch.Tensor, shared_biases: torch.Tensor) -> torch.Tensor:
        output_bias = torch.cat((self.output_bias, shared_biases.reshape(-1)))
        hidden = self.fc1(x) + output_bias.mean()
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        return F.pad(output, (0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
=======
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
>>>>>>> REPLACE

<<<<<<< SEARCH
            normalized_shared_bias_column = self.blocks[-1].attn.proj.weight[:, 2]
            normalized_shared_bias_column.sub_(
                normalized_shared_bias_column.mean()
            )
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
=======
            normalized_shared_bias_column = self.blocks[-1].attn.proj.weight[:, 2]
            normalized_shared_bias_column.sub_(
                normalized_shared_bias_column.mean()
            )
            normalized_final_scale_column = self.blocks[-1].attn.proj.weight[:, 3]
            normalized_final_scale_column.add_(
                target_mean - normalized_final_scale_column.mean()
            )
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        x = F.embedding(idx, self.token_weight()) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        token_weight = self.token_weight()
        x = F.embedding(idx, token_weight) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        final_proj_weight = self.blocks[-1].attn.proj.weight
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
            )
        )
        shared_final_bias = (
            math.sqrt(self.cfg.d_model) * final_proj_weight[:, 2].mean()
        )
        x = self.ln_f(x, shared_final_scales, shared_final_bias)
        logits = F.linear(x, self.token_weight())
=======
        final_proj_weight = self.blocks[-1].attn.proj.weight
        normalized_shared_values = (
            math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1:4].mean(dim=0)
        )
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                normalized_shared_values[0],
                normalized_shared_values[2],
            )
        )
        shared_final_bias = normalized_shared_values[1]
        x = self.ln_f(x, shared_final_scales, shared_final_bias)
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE