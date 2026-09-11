MECHANISM: Global tied-embedding quotient under pre-LayerNorm and logit-shift invariance

HYPOTHESIS: Extending the qualified positional-plus-MLP quotient design with removal of the tied token embedding’s single globally uniform degree will produce a 1611-parameter model with at least 99% accuracy.

INTENDED_EDIT: Compress each positional row and the tied token embedding into relative coordinates, reconstruct them for embedding and logit computation, and preserve their full-coordinate clipping and AdamW difference dynamics alongside the MLP bias quotient.

EVIDENCE: The 1612-parameter positional-plus-MLP quotient design achieved 99.82%. The failed 1611 design additionally quotienting the attention-output bias indicates that bias interaction should remain untouched; this patch instead removes a distinct exact gauge while retaining the full attention projection bias.

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # A global feature-uniform shift of every tied token row is canceled
        # at the input by pre-LayerNorm and becomes a vocabulary-uniform logit
        # shift at the output. Retain only relative flattened coordinates.
        full_token_weight = self.token_emb.weight.detach().flatten()
        self.token_emb.weight = nn.Parameter(
            full_token_weight[:-1] - full_token_weight[-1]
        )
        self.lm_head.weight = self.token_emb.weight

        # Feature-uniform offsets in individual positional rows are canceled
        # by all downstream LayerNorms. Retain relative coordinates per row.
        full_pos_weight = self.pos_emb.weight.detach()
        self.pos_emb.weight = nn.Parameter(
            full_pos_weight[:, :-1] - full_pos_weight[:, -1:]
        )

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)

        token_relative = torch.cat(
            (
                self.token_emb.weight,
                self.token_emb.weight.new_zeros(1),
            )
        )
        token_weight = (
            token_relative + self.token_emb.weight.mean()
        ).view(self.cfg.vocab_size, self.cfg.d_model)

        pos_relative = torch.cat(
            (
                self.pos_emb.weight,
                self.pos_emb.weight.new_zeros(
                    (self.pos_emb.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        pos_weight = pos_relative + self.pos_emb.weight.mean(
            dim=-1, keepdim=True
        )

        x = F.embedding(idx, token_weight) + F.embedding(pos, pos_weight)
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class QuotientAdamW(torch.optim.AdamW):
    """AdamW preserving the omitted uniform-bias coordinate's dynamics."""
=======
class QuotientAdamW(torch.optim.AdamW):
    """AdamW preserving omitted uniform coordinates along the last axis."""
>>>>>>> REPLACE

<<<<<<< SEARCH
            if "quotient_step" not in state:
                state["quotient_step"] = 0
                state["quotient_exp_avg"] = param.new_zeros(param.numel() + 1)
                state["quotient_exp_avg_sq"] = param.new_zeros(param.numel() + 1)

            full_grad = torch.cat((grad, -grad.sum().reshape(1)))
=======
            if "quotient_step" not in state:
                full_shape = list(param.shape)
                full_shape[-1] += 1
                state["quotient_step"] = 0
                state["quotient_exp_avg"] = param.new_zeros(full_shape)
                state["quotient_exp_avg_sq"] = param.new_zeros(full_shape)

            full_grad = torch.cat(
                (grad, -grad.sum(dim=-1, keepdim=True)),
                dim=-1,
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            param.add_(
                full_update[:-1] - full_update[-1],
                alpha=-step_size,
            )
=======
            param.add_(
                full_update[..., :-1] - full_update[..., -1:],
                alpha=-step_size,
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for param in parameters:
        if param.grad is None:
            continue
        term = param.grad.detach().square().sum()
        if id(param) in quotient_ids:
            term = term + param.grad.detach().sum().square()
        total_sq = term if total_sq is None else total_sq + term
=======
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        term = grad.square().sum()
        if id(param) in quotient_ids:
            term = term + grad.sum(dim=-1).square().sum()
        total_sq = term if total_sq is None else total_sq + term
>>>>>>> REPLACE

<<<<<<< SEARCH
    quotient_params = [block.mlp.fc2.bias for block in model.blocks]
=======
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.mlp.fc2.bias for block in model.blocks
    ]
>>>>>>> REPLACE