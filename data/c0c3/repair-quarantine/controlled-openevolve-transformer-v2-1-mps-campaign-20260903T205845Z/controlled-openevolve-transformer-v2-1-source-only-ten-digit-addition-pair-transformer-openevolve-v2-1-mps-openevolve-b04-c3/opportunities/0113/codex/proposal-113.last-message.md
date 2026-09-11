MECHANISM: Hierarchical tied-embedding rotational gauge fixing

HYPOTHESIS: Fixing one coordinate of the penultimate tied token vector will retain at least 99% accuracy after 45,000 updates while reducing the verified model from 1,138 to 1,137 learned parameters.

INTENDED_EDIT: Preserve the final token’s learned anchor, remove the penultimate token’s last coordinate, and reconstruct both constrained vectors during the forward pass.

EVIDENCE: The current design achieved 1.0 accuracy after fixing five coordinates of the final tied token vector, while extending the attention-output trim failed at 0.9382; this motivates continuing along the remaining token-subspace rotational gauge instead.

<<<<<<< SEARCH
        # Complete the rotational basis choice for the final tied token vector
        # by fixing five of its six coordinates. Constructing the full
        # Embedding above preserves constructor RNG consumption.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 5)
        )
        self.token_emb.fixed_weight_trim = (0, 5)
=======
        # The final tied vector fixes five rotational degrees. Use one remaining
        # rotation orthogonal to it to fix the penultimate vector's last
        # coordinate while retaining the final vector's learned anchor.
        # Constructing the full Embedding preserves constructor RNG consumption.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 6)
        )
        self.token_emb.fixed_weight_trim = (0, 5)
        self.token_emb.fix_penultimate_tail = True
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, nn.Embedding):
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.num_embeddings * module.embedding_dim
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                start, end_trim = module.fixed_weight_trim
                end = full_weight.numel() - end_trim
                with torch.no_grad():
                    module.weight.copy_(full_weight[start:end])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
=======
        elif isinstance(module, nn.Embedding):
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.num_embeddings * module.embedding_dim
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                if hasattr(module, "fix_penultimate_tail"):
                    last_token_start = (
                        full_weight.numel() - module.embedding_dim
                    )
                    compact_weight = torch.cat(
                        (
                            full_weight[: last_token_start - 1],
                            full_weight[last_token_start : last_token_start + 1],
                        )
                    )
                else:
                    start, end_trim = module.fixed_weight_trim
                    end = full_weight.numel() - end_trim
                    compact_weight = full_weight[start:end]
                with torch.no_grad():
                    module.weight.copy_(compact_weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_weight = F.pad(self.token_emb.weight, (0, 5)).view(
            self.cfg.vocab_size,
            self.token_dim,
        )
=======
        token_weight = torch.cat(
            (
                self.token_emb.weight[:-1],
                self.token_emb.weight.new_zeros(1),
                self.token_emb.weight[-1:],
                self.token_emb.weight.new_zeros(5),
            )
        ).view(
            self.cfg.vocab_size,
            self.token_dim,
        )
>>>>>>> REPLACE