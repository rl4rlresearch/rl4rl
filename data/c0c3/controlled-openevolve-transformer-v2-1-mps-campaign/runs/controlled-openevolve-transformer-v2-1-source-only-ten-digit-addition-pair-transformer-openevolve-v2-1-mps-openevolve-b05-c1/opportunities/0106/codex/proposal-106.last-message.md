MECHANISM: Exact value/output rotation-gauge quotient

HYPOTHESIS: Removing one redundant value/output rotation coefficient will produce a 1,265-parameter model with at least 99% accuracy, while cached tied weights and zero-dropout identities offset reconstruction overhead sufficiently for verification to finish.

INTENDED_EDIT: Canonicalize one first-head value coefficient to zero, inverse-rotate the matching output-projection columns at initialization, store only the remaining value coefficients, and streamline zero-dropout and tied-weight use.

EVIDENCE: The 1,266-parameter design achieved 100% accuracy. The prior value/output-gauge reuse was unverifiable because it coupled the gauge coefficient to the final LayerNorm scale; directly quotienting the same exact symmetry while retaining that scale isolates the parameter reduction, and avoids the repeatedly timed-out extra column-mean reduction.

<<<<<<< SEARCH
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
        self.v_weight = nn.Parameter(torch.empty(d_model, in_features))
=======
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
        self.in_features = in_features
        self.n_value_rows = d_model
        self.v_weight = nn.Parameter(torch.empty(d_model * in_features - 1))
        self.register_buffer(
            "value_rotation",
            torch.eye(2),
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2.copy_(second_coeff[3:4])
            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)
=======
            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2.copy_(second_coeff[3:4])

            dense_value = self.v_weight.new_empty(
                self.n_value_rows, self.in_features
            )
            nn.init.normal_(dense_value, mean=0.0, std=0.02)
            first_value = dense_value[0].clone()
            second_value = dense_value[1].clone()
            radius = torch.sqrt(
                first_value[0].square() + second_value[0].square()
            )
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = second_value[0] / radius
            sine = -first_value[0] / radius
            dense_value[0] = cosine * first_value + sine * second_value
            dense_value[1] = -sine * first_value + cosine * second_value
            dense_value[0, 0] = 0.0
            self.v_weight.copy_(dense_value.reshape(-1)[1:])
            self.value_rotation.copy_(
                torch.stack(
                    (
                        torch.stack((cosine, sine)),
                        torch.stack((-sine, cosine)),
                    )
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_head = self.head_basis @ second_head_coeff
        qk_weight = torch.cat((first_head, second_head), dim=0)
        return F.linear(
            x[..., :-1],
            torch.cat((qk_weight, self.v_weight), dim=0),
        )
=======
        second_head = self.head_basis @ second_head_coeff
        qk_weight = torch.cat((first_head, second_head), dim=0)
        v_weight = F.pad(self.v_weight, (1, 0)).view(
            self.n_value_rows, self.in_features
        )
        return F.linear(
            x[..., :-1],
            torch.cat((qk_weight, v_weight), dim=0),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
=======
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)
        self.resid_drop = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))
        self.drop = nn.Dropout(dropout)
=======
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))
        self.drop = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
=======
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = (
            nn.Identity() if cfg.dropout == 0.0 else nn.Dropout(cfg.dropout)
        )
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
        with torch.no_grad():
            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
=======
        self.apply(self._init_weights)
        with torch.no_grad():
            for block in self.blocks:
                rotation = block.attn.qkv.value_rotation
                value_columns = block.attn.proj.weight[:, :2].clone()
                block.attn.proj.weight[:, :2].copy_(
                    value_columns @ rotation.transpose(0, 1)
                )

            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        x = F.embedding(idx, self.token_weight()) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        token_weight = self.token_weight()
        x = F.embedding(idx, token_weight) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x, shared_final_scales, shared_final_bias)
        logits = F.linear(x, self.token_weight())
=======
        x = self.ln_f(x, shared_final_scales, shared_final_bias)
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE