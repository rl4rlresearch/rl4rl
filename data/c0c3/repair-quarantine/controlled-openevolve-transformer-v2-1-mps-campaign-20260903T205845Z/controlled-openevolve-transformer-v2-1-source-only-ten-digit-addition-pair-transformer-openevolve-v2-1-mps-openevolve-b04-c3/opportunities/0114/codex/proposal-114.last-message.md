MECHANISM: Hierarchical tied-embedding rotational gauge fixing

HYPOTHESIS: Anchoring the final token to one learned coordinate and fixing one trailing coordinate of the penultimate token will reduce the model to 1,137 parameters while retaining at least 99% accuracy after 45,000 updates.

INTENDED_EDIT: Replace the contiguous four-scalar embedding trim with a compact representation that omits five coordinates from the final token and one from the penultimate token.

EVIDENCE: The 1,138-parameter reference reached 1.0 accuracy with the final token anchored to one coordinate, while the prior 1,137-parameter hierarchical test timed out without producing adverse accuracy evidence; this patch tests that remaining stabilizer gauge with lightweight reconstruction.

<<<<<<< SEARCH
        # Choose a basis for the tied token subspace by fixing its final four
        # scalars. Constructing the full Embedding above preserves constructor RNG.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 4)
        )
        self.token_emb.fixed_weight_trim = (0, 4)
=======
        # Anchor the final tied token to one coordinate, then use the residual
        # rotational freedom to fix the penultimate token's last coordinate.
        # Constructing the full Embedding above preserves constructor RNG.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 6)
        )
        self.token_emb.hierarchical_gauge = True
>>>>>>> REPLACE

<<<<<<< SEARCH
                start, end_trim = module.fixed_weight_trim
                end = full_weight.numel() - end_trim
                with torch.no_grad():
                    module.weight.copy_(full_weight[start:end])
=======
                if hasattr(module, "hierarchical_gauge"):
                    full_weight = full_weight.view(
                        module.num_embeddings,
                        module.embedding_dim,
                    )
                    compact_weight = torch.cat(
                        (
                            full_weight[:-2].flatten(),
                            full_weight[-2, :-1],
                            full_weight[-1, :1],
                        )
                    )
                    with torch.no_grad():
                        module.weight.copy_(compact_weight)
                else:
                    start, end_trim = module.fixed_weight_trim
                    end = full_weight.numel() - end_trim
                    with torch.no_grad():
                        module.weight.copy_(full_weight[start:end])
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_weight = F.pad(self.token_emb.weight, (0, 4)).view(
            self.cfg.vocab_size,
            self.token_dim,
        )
=======
        token_prefix, penultimate_token, final_token = torch.split(
            self.token_emb.weight,
            (
                (self.cfg.vocab_size - 2) * self.token_dim,
                self.token_dim - 1,
                1,
            ),
        )
        token_weight = torch.cat(
            (
                token_prefix,
                F.pad(penultimate_token, (0, 1)),
                F.pad(final_token, (0, self.token_dim - 1)),
            )
        ).view(self.cfg.vocab_size, self.token_dim)
>>>>>>> REPLACE