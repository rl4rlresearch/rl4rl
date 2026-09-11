MECHANISM: Complete first-head centered rotation gauge with hot-path streamlining

HYPOTHESIS: Fixing the remaining exact centered-subspace rotation in the first attention head will reduce the model from 1,266 to 1,265 parameters while retaining at least 99% accuracy; eliminating zero-dropout dispatch, duplicate reductions, and redundant token-weight reconstruction will help the previously timed-out gauge reduction finish verification.

INTENDED_EDIT: Rotate the first head’s center-1/center-2 plane so the center-2 coefficient at input coordinate 1 is zero, omit that coefficient, and streamline mathematically equivalent zero-dropout, shared-bias, normalization-statistic, token-weight, and masked-loss operations.

EVIDENCE: The 1,266-parameter design reached 100%, and two existing first-head rotation constraints also retained 100%; the same final exact gauge constraint previously timed out without contrary accuracy evidence, while failures from constraining the second head make completing the already-tolerant first-head gauge the better-supported reduction.

<<<<<<< SEARCH
class SingleRotationGaugeQKV(nn.Module):
    """Tied query/key and value map with one centered rotation fixed per head."""
=======
class SingleRotationGaugeQKV(nn.Module):
    """Tied query/key and value map with centered rotation gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qk_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_center2_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_common = nn.Parameter(torch.empty(1, in_features))
=======
        self.qk_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_center2_tail = nn.Parameter(torch.empty(1, in_features - 2))
        self.qk_second_common = nn.Parameter(torch.empty(1, in_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
            coeff[1] = cosine * first + sine * second
            coeff[3] = -sine * first + cosine * second

            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2_tail.copy_(coeff[3:4, 1:])
=======
            coeff[1] = cosine * first + sine * second
            coeff[3] = -sine * first + cosine * second

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
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qk_common,
                self.qk_center0,
                F.pad(self.qk_center1_tail, (1, 0)),
                F.pad(self.qk_center2_tail, (1, 0)),
=======
                self.qk_common,
                self.qk_center0,
                F.pad(self.qk_center1_tail, (1, 0)),
                F.pad(self.qk_center2_tail, (2, 0)),
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
=======
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        qk, v = qkv.chunk(2, dim=-1)

        q = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = q + self.proj.bias.mean()
=======
    def forward(
        self, x: torch.Tensor, shared_query_bias: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        qk, v = qkv.chunk(2, dim=-1)

        q = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = q + shared_query_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y
=======
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        if self.dropout:
            att = F.dropout(att, p=self.dropout, training=self.training)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        if self.dropout:
            y = F.dropout(y, p=self.dropout, training=self.training)
        return y
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))
        self.drop = nn.Dropout(dropout)
=======
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))
        self.dropout = dropout
>>>>>>> REPLACE

<<<<<<< SEARCH
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        output = F.pad(output, (0, 1))
        return self.drop(output)
=======
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        output = F.pad(output, (0, 1))
        if self.dropout:
            output = F.dropout(output, p=self.dropout, training=self.training)
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
        shared_bias_mean = self.attn.proj.bias.mean()
        shared_biases = torch.stack(
            (shared_bias_mean, self.attn.proj.bias[0])
        )
        x = x + self.attn(self.ln1(x), shared_bias_mean)
        x = x + self.mlp(self.ln2(x), shared_biases)
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
=======
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.dropout = cfg.dropout
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        x = F.embedding(idx, self.token_weight()) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

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

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        token_weight = self.token_weight()
        x = F.embedding(idx, token_weight) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)

        for blk in self.blocks:
            x = blk(x)

        final_proj_weight = self.blocks[-1].attn.proj.weight
        shared_final = final_proj_weight[:, :3].mean(dim=0)
        shared_final_scales = torch.stack(
            (
                shared_final[0],
                math.sqrt(self.cfg.d_model) * shared_final[1],
            )
        )
        shared_final_bias = math.sqrt(self.cfg.d_model) * shared_final[2]
        x = self.ln_f(x, shared_final_scales, shared_final_bias)
        logits = F.linear(x, token_weight)

        loss = None
        if targets is not None:
            flat_targets = targets.reshape(-1)
            active = flat_targets != -100
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1))[active],
                flat_targets[active],
            )
>>>>>>> REPLACE