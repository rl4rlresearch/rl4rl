MECHANISM: Sixth terminal-column output-shift gauge with ambient AdamW

HYPOTHESIS: A 1,532-parameter model will retain at least 99% accuracy because the qualified 1,533-parameter design reached 99.89%, while a sixth `fc2` column’s common-output component is also exactly erased by the final LayerNorm and its full AdamW dynamics can be preserved.

INTENDED_EDIT: Reproduce the qualified tied-token gauge, five absorbed `ln2` scales, and five terminal-column gauges, then gauge-fix a sixth terminal column using the existing ambient-coordinate optimizer.

EVIDENCE: The five-terminal-column 1,533-parameter design achieved 99.89% accuracy; extending the same exact symmetry by one column is the smallest controlled reduction beyond the best qualified design.

<<<<<<< SEARCH
class GaugeFixedPositionEmbedding(nn.Module):
=======
class GaugeFixedTokenEmbedding(nn.Module):
    """Tied embedding with its global scalar-shift gauge removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.weight.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        flat = raw.reshape(-1)
        self.weight.copy_(flat[:-1] - flat[-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (self.weight, self.weight.new_zeros(1))
        ).view(self.num_embeddings, self.embedding_dim)
        if torch.is_grad_enabled():
            full_weight.retain_grad()
        self.full_weight = full_weight
        return F.embedding(idx, full_weight)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight)


class TiedTokenProjection(nn.Module):
    """Parameter-free output view of the learned token embedding."""

    def __init__(self, embedding: GaugeFixedTokenEmbedding):
        super().__init__()
        self.in_features = embedding.embedding_dim
        self.out_features = embedding.num_embeddings
        object.__setattr__(self, "embedding", embedding)

        # Match the constructor RNG consumption of the replaced nn.Linear.
        scratch = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(scratch, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding.project(x)


class GaugeFixedPositionEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and four weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(4)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 4)
        )
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and six weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(6)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 6)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 6:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
class OneFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with one scale absorbed by the following linear."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
=======
class FiveFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with five scales absorbed by following columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = OneFixedScaleLayerNorm(cfg.d_model)
=======
        self.ln2 = FiveFixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Parameter-free output view preserves input/output weight tying.
        self.lm_head = TiedTokenProjection(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedTokenEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, TiedTokenProjection):
            # Match the second initialization of the formerly tied Linear.
            module.embedding.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
>>>>>>> REPLACE

<<<<<<< SEARCH
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
=======
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 6:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # attention-weight, attention-bias, terminal-bias, and four
    # terminal-weight gauges.
    gauge_params = [model.pos_emb.first]
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, attention-weight, attention-bias, terminal-bias, and six
    # terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_step = 0

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    gauge_step = 0

    # The last five ln2 scales are redundant with the corresponding fc1
    # columns. Keep their factorization only as optimizer-coordinate state,
    # while the model stores and uses the deduplicated effective columns.
    absorbed_scales = [
        torch.ones(5, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -5:])
        for blk in model.blocks
    ]
    absorbed_weight_v = [
        torch.zeros_like(moment) for moment in absorbed_weight_m
    ]
    absorbed_scale_m = [
        torch.zeros_like(scale) for scale in absorbed_scales
    ]
    absorbed_scale_v = [
        torch.zeros_like(moment) for moment in absorbed_scale_m
    ]
    absorbed_step = 0

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
=======
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.mlp.fc2.full_weight_prefix
            )

        clip_scale = 1.0
=======
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.mlp.fc2.full_weight_prefix
            )

        absorbed_grads = []
        for blk, virtual_scale in zip(model.blocks, absorbed_scales):
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -5:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -5:].detach()
                / virtual_scale.unsqueeze(0)
            )
            ambient_weight_grad = (
                effective_grad * virtual_scale.unsqueeze(0)
            )
            ambient_scale_grad = (
                effective_grad * virtual_weight
            ).sum(dim=0)
            absorbed_grads.append(
                (
                    effective_grad,
                    virtual_weight,
                    ambient_weight_grad,
                    ambient_scale_grad,
                )
            )

        clip_scale = 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            total_norm = float(grad_sq.sqrt().item())
=======
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for (
                effective_grad,
                _,
                ambient_weight_grad,
                ambient_scale_grad,
            ) in absorbed_grads:
                grad_sq = (
                    grad_sq
                    - effective_grad.float().square().sum()
                    + ambient_weight_grad.float().square().sum()
                    + ambient_scale_grad.float().square().sum()
                )
            total_norm = float(grad_sq.sqrt().item())
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()

        gauge_step += 1
=======
        # These effective columns are updated below through their ambient
        # weight/scale factorization rather than by the ordinary optimizer.
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -5:].zero_()

        optimizer.step()

        absorbed_step += 1
        absorbed_bc1 = 1.0 - 0.9 ** absorbed_step
        absorbed_bc2 = 1.0 - 0.999 ** absorbed_step
        for i, (blk, absorbed_grad) in enumerate(
            zip(model.blocks, absorbed_grads)
        ):
            (
                _,
                virtual_weight,
                ambient_weight_grad,
                ambient_scale_grad,
            ) = absorbed_grad
            weight_grad = ambient_weight_grad * clip_scale
            scale_grad = ambient_scale_grad * clip_scale
            weight_moment = absorbed_weight_m[i]
            weight_variance = absorbed_weight_v[i]
            scale_moment = absorbed_scale_m[i]
            scale_variance = absorbed_scale_v[i]
            virtual_scale = absorbed_scales[i]

            weight_moment.mul_(0.9).add_(weight_grad, alpha=0.1)
            weight_variance.mul_(0.999).addcmul_(
                weight_grad, weight_grad, value=0.001
            )
            scale_moment.mul_(0.9).add_(scale_grad, alpha=0.1)
            scale_variance.mul_(0.999).addcmul_(
                scale_grad, scale_grad, value=0.001
            )
            weight_direction = (
                weight_moment / absorbed_bc1
            ) / (
                (weight_variance / absorbed_bc2).sqrt() + 1e-8
            )
            scale_direction = (
                scale_moment / absorbed_bc1
            ) / (
                (scale_variance / absorbed_bc2).sqrt() + 1e-8
            )

            with torch.no_grad():
                decay = 1.0 - lr_now * train_cfg.weight_decay
                virtual_weight.mul_(decay).add_(
                    weight_direction, alpha=-lr_now
                )
                virtual_scale.mul_(decay).add_(
                    scale_direction, alpha=-lr_now
                )
                blk.mlp.fc1.weight[:, -5:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )

        gauge_step += 1
>>>>>>> REPLACE