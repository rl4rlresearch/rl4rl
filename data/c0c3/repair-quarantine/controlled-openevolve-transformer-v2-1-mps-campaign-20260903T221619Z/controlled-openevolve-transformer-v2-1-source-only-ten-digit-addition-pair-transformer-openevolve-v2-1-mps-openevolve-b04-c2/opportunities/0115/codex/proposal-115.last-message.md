MECHANISM: Progressive downstream-affine LayerNorm bias quotient

HYPOTHESIS: Reproducing the qualified vocabulary-centering design while reducing `ln2.bias` from five to four orthonormal coordinates will produce a 1,565-parameter model with at least 99% accuracy.

INTENDED_EDIT: Center the tied token embedding across vocabulary, transfer its mean into positional embeddings, use full-coordinate AdamW updates for the compact embedding, and remove one additional `ln2.bias` direction beyond the qualified 1,566-parameter design.

EVIDENCE: Vocabulary centering achieved 100% accuracy at 1,568 parameters, while successive `ln2.bias` reductions achieved 99.90% at 1,567 and 99.88% at 1,566; the next reduction tests the same downstream-bias redundancy without adding a previously failed value, query-key, or positional constraint.

<<<<<<< SEARCH
        # A uniform component of this post-normalization bias is absorbable
        # by the unrestricted fc1 bias. Remove only that single direction.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 1))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
=======
        # Constant post-normalization shifts are absorbable by the
        # unrestricted fc1 bias. Retain four coordinates to test the next
        # progressive quotient beyond the qualified five-coordinate design.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 4))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 4)
        for col in range(cfg.d_model - 4):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 21
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (
                    initialized_pos[:7],
                    initialized_pos[8:13],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[13:-9]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    unrestricted_pos.reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
        # Remove the vocabulary-common component of the tied token embedding.
        # Its output contribution is a common logit shift, while transferring
        # its mean to every positional row preserves the model input.
        token_basis = torch.zeros(cfg.vocab_size, cfg.vocab_size - 1)
        for col in range(cfg.vocab_size - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            token_basis[: col + 1, col] = 1.0 / scale
            token_basis[col + 1, col] = -(col + 1) / scale

        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 21
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
            selected_pos = torch.cat(
                (
                    initialized_pos[:7],
                    initialized_pos[8:13],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[13:-9]), dim=0
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        token_weight = self.token_basis @ self.token_emb.weight
        x = F.embedding(idx, token_weight) + F.embedding(pos, pos_weight)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_params = [
        pair
        for blk in model.blocks
=======
    gauge_params = [(model.token_emb.weight, model.token_basis)] + [
        pair
        for blk in model.blocks
>>>>>>> REPLACE