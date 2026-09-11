MECHANISM: Sparse horizon-2 auxiliary prediction

HYPOTHESIS: On the best 2.25×/2.75×/5.5× MLP taper, a separate horizon-2 head trained on every fourth position with weight 0.2 will retain at least 490M tokens and reduce val_bpb below 0.991682.

INTENDED_EDIT: Restore the best verified MLP allocation and challenge the assumption that next-token supervision alone produces the best context representation; training adds sparse token-t+2 prediction while validation and primary predictions remain unchanged.

EVIDENCE: The 2.25×/2.75×/5.5× design achieved 0.991682 at 510.7M tokens, while neighboring width reallocations and SwiGLU did not improve it; preserving ReLU² and introducing a distinct prediction horizon tests a new mechanism with limited compute overhead.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if layer_idx < config.n_layer // 4:
            mlp_mult_halves = 4   # 2x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_halves = 6   # 3x
        else:
            mlp_mult_halves = 11  # 5.5x
        mlp_dim = mlp_mult_halves * config.n_embd // 2
=======
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if layer_idx < config.n_layer // 4:
            mlp_mult_quarters = 9   # 2.25x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_quarters = 11  # 2.75x
        else:
            mlp_mult_quarters = 22  # 5.5x
        mlp_dim = mlp_mult_quarters * config.n_embd // 4
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
=======
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.future_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
>>>>>>> REPLACE

<<<<<<< SEARCH
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
=======
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        torch.nn.init.normal_(self.future_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
=======
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        future_head_numel = self.future_head.weight.numel()
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        base_compute_numel = nparams - nparams_exclude - future_head_numel
        h = self.config.n_head
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 6 * (nparams - nparams_exclude) + attn_flops
=======
        # The auxiliary vocabulary head is evaluated on one quarter of positions.
        return 6 * base_compute_numel + 6 * future_head_numel / 4 + attn_flops
>>>>>>> REPLACE

<<<<<<< SEARCH
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }
=======
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        future_head = sum(p.numel() for p in self.future_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + future_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'future_head': future_head, 'transformer_matrices': transformer_matrices,
            'scalars': scalars, 'total': total,
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
=======
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        future_head_params = list(self.future_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(future_head_params) + len(value_embeds_params) +
            len(resid_params) + len(x0_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=lm_head_params + future_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits
=======
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            if self.training:
                # From context ending at token t, separately predict token t+2.
                future_x = x[:, :-1:4]
                future_targets = targets[:, 1::4]
                future_logits = self.future_head(future_x).float()
                future_logits = softcap * torch.tanh(future_logits / softcap)
                future_loss = F.cross_entropy(
                    future_logits.reshape(-1, future_logits.size(-1)),
                    future_targets.reshape(-1), ignore_index=-1, reduction=reduction,
                )
                loss = loss + 0.2 * future_loss
            return loss
        return logits
>>>>>>> REPLACE