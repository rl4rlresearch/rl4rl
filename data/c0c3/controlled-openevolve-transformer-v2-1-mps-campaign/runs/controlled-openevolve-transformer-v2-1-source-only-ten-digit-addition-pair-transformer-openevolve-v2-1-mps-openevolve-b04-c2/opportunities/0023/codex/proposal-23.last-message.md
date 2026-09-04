MECHANISM: Final-LayerNorm output-scale gauge fixing

HYPOTHESIS: Building on the verified 1,621-parameter design, fixing the mean final-LayerNorm gain to one will yield a 1,620-parameter model with at least 99% accuracy because positive global scaling of its gain and bias cannot change greedy argmax decoding.

INTENDED_EDIT: Apply the verified value-bias removal and final-four positional quotient, then represent the final LayerNorm gain as seven learned zero-mean deviations around a fixed unit mean.

EVIDENCE: The value-bias-free, final-four-position design achieved 99.93% accuracy at 1,621 parameters; unlike the failed fifth positional-row and hidden LayerNorm-bias reductions, this removes only a global output scale that is irrelevant to the protected decoder’s argmax.

<<<<<<< SEARCH
        # A shared key bias cancels from every attention-softmax row. Retain
        # the original parameter slot while storing only query/value biases.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
=======
        # A shared key bias cancels from every attention-softmax row. A value
        # bias passes unchanged through attention and is absorbable by the
        # output-projection bias, so retain only the learned query bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
=======
        q_bias = self.qkv.bias
        zero_bias = torch.zeros_like(q_bias)
        qkv_bias = torch.cat((q_bias, zero_bias, zero_bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final three positions, which have the shortest causal influence.
        self.compact_pos_count = 5
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final four positions, which have the shortest causal influence.
        self.compact_pos_count = 6
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-3:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-3].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-4:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-4].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)

        # Positive common scaling of the final affine LayerNorm output leaves
        # greedy argmax decoding unchanged. Fix the mean gain to one and learn
        # only its seven zero-sum deviations.
        self.ln_f.weight = nn.Parameter(
            self.ln_f.weight.new_zeros(cfg.d_model - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        ln_f_weight = (
            torch.ones_like(self.ln_f.bias)
            + self.pos_basis @ self.ln_f.weight
        )
        x = F.layer_norm(
            x,
            (self.cfg.d_model,),
            ln_f_weight,
            self.ln_f.bias,
            self.ln_f.eps,
        )
        logits = self.lm_head(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-3:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-3].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-4:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-4].reshape(-1),
                    )
                )
>>>>>>> REPLACE