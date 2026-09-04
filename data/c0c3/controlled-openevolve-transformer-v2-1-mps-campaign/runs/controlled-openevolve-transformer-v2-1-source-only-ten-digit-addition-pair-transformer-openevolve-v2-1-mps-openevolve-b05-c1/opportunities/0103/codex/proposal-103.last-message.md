MECHANISM: Rotational-gauge parameter transfer with hot-path streamlining

HYPOTHESIS: Quotienting the remaining first-head centered rotation and reallocating its redundant parameter slot to the final LayerNorm scale will produce a 1,265-parameter model with at least 99% accuracy; reusing the projection-bias mean, skipping zero-dropout calls, caching token weights, and compacting ignored loss rows will let verification finish within the time limit.

INTENDED_EDIT: Fix the first head’s center-1/center-2 rotational gauge at input coordinate 1, replace the dedicated final-LayerNorm scale with a scalar stored in the freed QKV parameter budget, and reduce function-preserving training overhead.

EVIDENCE: The 1,266-parameter model reached 100% accuracy, and the identical first-head gauge reduction previously timed out without contrary accuracy evidence. Successful orthonormal null reuse at 1,267 and 1,266 shows that transferring a redundant degree of freedom to a required final-normalization parameter can preserve accuracy.

<<<<<<< SEARCH
        self.qk_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_center2_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_common = nn.Parameter(torch.empty(1, in_features))
=======
        self.qk_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_center2_tail = nn.Parameter(torch.empty(1, in_features - 2))
        self.final_scale = nn.Parameter(torch.ones(()))
        self.qk_second_common = nn.Parameter(torch.empty(1, in_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2_tail.copy_(coeff[3:4, 1:])

            dense_second = torch.empty_like(
=======
            first = coeff[2].clone()
            second = coeff[3].clone()
            radius = torch.sqrt(first[1].square() + second[1].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[1] / radius
            sine = second[1] / radius
            coeff[2] = cosine * first + sine * second
            coeff[3] = -sine * first + cosine * second

            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2_tail.copy_(coeff[3:4, 2:])
            self.final_scale.fill_(1.0)

            dense_second = torch.empty_like(
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.qk_center1_tail, (1, 0)),
                F.pad(self.qk_center2_tail, (1, 0)),
=======
                F.pad(self.qk_center1_tail, (1, 0)),
                F.pad(self.qk_center2_tail, (2, 0)),
>>>>>>> REPLACE

<<<<<<< SEARCH
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
=======
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        weight = torch.cat(
            (
                shared_scales[:1],
                shared_scales.new_ones(1),
                shared_scales[2:3],
                shared_scales.new_ones(2),
                shared_scales[1:2],
                shared_scales.new_ones(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
=======
    def forward(
        self, x: torch.Tensor, shared_q_bias: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = q + self.proj.bias.mean()
=======
        q = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = q + shared_q_bias
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
        if self.attn_drop.p:
            att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        if self.resid_drop.p:
            y = self.resid_drop(y)
        return y
>>>>>>> REPLACE

<<<<<<< SEARCH
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        output = F.pad(output, (0, 1))
        return self.drop(output)
=======
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        output = F.pad(output, (0, 1))
        if self.drop.p:
            output = self.drop(output)
        return output
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_biases = torch.stack(
            (self.attn.proj.bias.mean(), self.attn.proj.bias[0])
        )
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x), shared_biases)
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_bias_mean = self.attn.proj.bias.mean()
        shared_biases = torch.stack(
            (proj_bias_mean, self.attn.proj.bias[0])
        )
        x = x + self.attn(self.ln1(x), proj_bias_mean)
        x = x + self.mlp(self.ln2(x), shared_biases)
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = F.embedding(idx, self.token_weight()) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
        x = self.drop(x)

        for blk in self.blocks:
=======
        token_weight = self.token_weight()
        x = F.embedding(idx, token_weight) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
        if self.drop.p:
            x = self.drop(x)

        for blk in self.blocks:
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
            )
        )
=======
        shared_final_scales = torch.stack(
            (
                self.blocks[-1].attn.qkv.final_scale,
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x, shared_final_scales, shared_final_bias)
        logits = F.linear(x, self.token_weight())

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
=======
        x = self.ln_f(x, shared_final_scales, shared_final_bias)
        logits = F.linear(x, token_weight)

        loss = None
        if targets is not None:
            flat_targets = targets.reshape(-1)
            valid = flat_targets != -100
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1))[valid],
                flat_targets[valid],
            )
>>>>>>> REPLACE