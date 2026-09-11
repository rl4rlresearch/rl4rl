MECHANISM: Optimizer-preserving tied-embedding common-mode gauge quotient

HYPOTHESIS: Removing the single global additive degree of freedom shared by every tied embedding coordinate will reduce the model from 1,447 to 1,446 parameters while retaining at least 99% accuracy, because it changes input residuals only by LayerNorm-invisible scalar shifts and output logits only by a softmax-invisible common shift.

INTENDED_EDIT: Store all but one flattened embedding coordinate as differences from a fixed reference, reconstruct the tied input/output weight dynamically, preserve the original initialization RNG stream, and include the omitted coordinate in quotient-aware AdamW moments and gradient clipping.

EVIDENCE: The 1,447-parameter model reached 99.97%, and optimizer-preserving gauge quotients successfully removed both MLP residual and relative-attention common modes where direct reparameterizations disrupted training; this applies the same proven optimization treatment to another exact model-wide invariance.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with one global all-entries offset fixed at zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )

        # Preserve the RNG stream of nn.Embedding's constructor.
        torch.empty(num_embeddings, embedding_dim).normal_()

    def initialize_from_full_normal(self) -> None:
        full_weight = self.weight.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        flat = full_weight.reshape(-1)
        with torch.no_grad():
            self.weight.copy_(flat[:-1] - flat[-1])

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedLMHead(nn.Module):
    """Parameter-free output view of the gauge-fixed tied embedding."""

    def __init__(self, embedding: GaugeFixedEmbedding):
        super().__init__()
        object.__setattr__(self, "embedding", embedding)

        # Preserve the RNG stream of the removed nn.Linear constructor.
        discarded = torch.empty(
            embedding.num_embeddings, embedding.embedding_dim
        )
        nn.init.kaiming_uniform_(discarded, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.full_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
=======
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Parameter-free weight tying with the input embedding.
        self.lm_head = GaugeFixedLMHead(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "_residual_gauge", False):
                full_weight = module.weight.new_empty(
                    module.out_features + 1, module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[:-1] - full_weight[-1:])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
        if isinstance(module, GaugeFixedEmbedding):
            module.initialize_from_full_normal()
        elif isinstance(module, GaugeFixedLMHead):
            module.embedding.initialize_from_full_normal()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "_residual_gauge", False):
                full_weight = module.weight.new_empty(
                    module.out_features + 1, module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[:-1] - full_weight[-1:])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (block.attn.rel_bias, 1) for block in model.blocks
        ]
=======
        self.gauge_specs = [
            (model.token_emb.weight, 0)
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (block.attn.rel_bias, 1) for block in model.blocks
        ]
>>>>>>> REPLACE