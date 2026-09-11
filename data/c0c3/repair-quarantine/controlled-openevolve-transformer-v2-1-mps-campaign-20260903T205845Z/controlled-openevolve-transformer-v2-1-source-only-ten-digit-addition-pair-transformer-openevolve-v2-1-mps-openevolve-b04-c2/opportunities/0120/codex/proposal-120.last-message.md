MECHANISM: Indexed positional common-mode quotient

HYPOTHESIS: Extending the verified 1,564-parameter design with an order-safe quotient of positional row 14 will produce a 1,563-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified token-centering, ninth value rotation, six-coordinate `ln1.bias`, and four-coordinate `ln2.bias` reductions; additionally compact positional row 14 using explicit index buffers for initialization, inference reconstruction, gradients, and AdamW updates.

EVIDENCE: The 1,564-parameter LayerNorm-bias design achieved 99.86% accuracy. Earlier row-14 attempts produced no accuracy measurement, so an indexed implementation that eliminates fragile manual row ordering is the most informative independent one-parameter reduction.

<<<<<<< SEARCH
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
=======
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
                elif head == 1:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # A uniform pre-attention shift is absorbable by the retained query
        # and output-projection biases; key shifts cancel from the softmax.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
=======
        # Retain the qualified six-coordinate pre-attention bias quotient.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 2))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # A uniform component of this post-normalization bias is absorbable
        # by the unrestricted fc1 bias. Remove only that single direction.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 1))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
=======
        # Retain the qualified four-coordinate post-normalization bias
        # quotient; omitted shifts are absorbable by the fc1 bias.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 4))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 4)
        for col in range(cfg.d_model - 4):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()
            block.mlp.compact_output_columns()

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of the first seven and final
        # nine positional rows.
        self.compact_pos_count = 16
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:7], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[7:-9].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()
            block.mlp.compact_output_columns()

        # Remove the vocabulary-common component of the tied embedding and
        # transfer its input contribution to every positional row.
        token_basis = torch.zeros(cfg.vocab_size, cfg.vocab_size - 1)
        for col in range(cfg.vocab_size - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            token_basis[: col + 1, col] = 1.0 / scale
            token_basis[col + 1, col] = -(col + 1) / scale

        # Compact rows 0-6, 8-12, 14, and the final nine rows. Explicit
        # indices keep packed storage and sequence order independent.
        compact_pos_indices = torch.cat(
            (
                torch.arange(0, 7),
                torch.arange(8, 13),
                torch.tensor([14]),
                torch.arange(cfg.max_seq_len - 9, cfg.max_seq_len),
            )
        )
        unrestricted_mask = torch.ones(cfg.max_seq_len, dtype=torch.bool)
        unrestricted_mask[compact_pos_indices] = False
        unrestricted_pos_indices = torch.arange(cfg.max_seq_len)[
            unrestricted_mask
        ]
        packed_pos_indices = torch.cat(
            (compact_pos_indices, unrestricted_pos_indices)
        )
        pos_unpack_indices = torch.argsort(packed_pos_indices)
        self.compact_pos_count = int(compact_pos_indices.numel())

        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale

        with torch.no_grad():
            initialized_token = self.token_emb.weight.detach()
            token_mean = initialized_token.mean(dim=0)
            compact_token = token_basis.transpose(0, 1) @ initialized_token

            initialized_pos = self.pos_emb.weight.detach() + token_mean
            selected_pos = initialized_pos.index_select(
                0, compact_pos_indices
            )
            unrestricted_pos = initialized_pos.index_select(
                0, unrestricted_pos_indices
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    unrestricted_pos.reshape(-1),
                )
            )

        self.lm_head = None
        self.token_emb.weight = nn.Parameter(compact_token)
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("token_basis", token_basis, persistent=False)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
        self.register_buffer(
            "pos_compact_indices", compact_pos_indices, persistent=False
        )
        self.register_buffer(
            "pos_unrestricted_indices",
            unrestricted_pos_indices,
            persistent=False,
        )
        self.register_buffer(
            "pos_unpack_indices", pos_unpack_indices, persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        compact_size = self.compact_pos_count * (self.cfg.d_model - 1)
        compact_rows = (
            self.pos_emb.weight[:compact_size].view(
                self.compact_pos_count, self.cfg.d_model - 1
            )
            @ self.pos_basis.transpose(0, 1)
        )
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[7:],
            ),
            dim=0,
        )
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        compact_size = self.compact_pos_count * (self.cfg.d_model - 1)
        compact_rows = (
            self.pos_emb.weight[:compact_size].view(
                self.compact_pos_count, self.cfg.d_model - 1
            )
            @ self.pos_basis.transpose(0, 1)
        )
        unrestricted_rows = self.pos_emb.weight[compact_size:].view(
            self.cfg.max_seq_len - self.compact_pos_count,
            self.cfg.d_model,
        )
        packed_rows = torch.cat(
            (compact_rows, unrestricted_rows), dim=0
        )
        pos_weight = packed_rows.index_select(0, self.pos_unpack_indices)
        token_weight = self.token_basis @ self.token_emb.weight
        x = F.embedding(idx, token_weight) + F.embedding(pos, pos_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_params = [
        pair
=======
    gauge_params = [(model.token_emb.weight, model.token_basis)] + [
        pair
>>>>>>> REPLACE

<<<<<<< SEARCH
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
=======
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
    pos_compact_indices = model.pos_compact_indices
    pos_unrestricted_indices = model.pos_unrestricted_indices
    pos_unpack_indices = model.pos_unpack_indices
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_full_grad = (
                    pos_param.grad[:pos_compact_size].view(
                        compact_pos_count, model_cfg.d_model - 1
                    )
                    @ pos_basis.transpose(0, 1)
                )
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[7:].reshape(-1),
                    )
                )
                state["exp_avg"].lerp_(full_grad, 1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (
                    state["exp_avg_sq"].sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(eps)
                )
                full_update = (
                    state["exp_avg"] / denom
                ).view(model_cfg.max_seq_len, model_cfg.d_model)
                selected_update = torch.cat(
                    (full_update[:7], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[7:-9].reshape(-1),
                    )
                )
                pos_param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                pos_param.add_(compact_update, alpha=-lr_now / bias_correction1)
=======
                selected_full_grad = (
                    pos_param.grad[:pos_compact_size].view(
                        compact_pos_count, model_cfg.d_model - 1
                    )
                    @ pos_basis.transpose(0, 1)
                )
                unrestricted_full_grad = pos_param.grad[
                    pos_compact_size:
                ].view(
                    model_cfg.max_seq_len - compact_pos_count,
                    model_cfg.d_model,
                )
                packed_full_grad = torch.cat(
                    (selected_full_grad, unrestricted_full_grad), dim=0
                )
                full_grad = packed_full_grad.index_select(
                    0, pos_unpack_indices
                ).reshape(-1)
                state["exp_avg"].lerp_(full_grad, 1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (
                    state["exp_avg_sq"].sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(eps)
                )
                full_update = (
                    state["exp_avg"] / denom
                ).view(model_cfg.max_seq_len, model_cfg.d_model)
                selected_update = full_update.index_select(
                    0, pos_compact_indices
                )
                unrestricted_update = full_update.index_select(
                    0, pos_unrestricted_indices
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        unrestricted_update.reshape(-1),
                    )
                )
                pos_param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                pos_param.add_(compact_update, alpha=-lr_now / bias_correction1)
>>>>>>> REPLACE