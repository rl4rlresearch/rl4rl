MECHANISM: Learned Gaussian delay-mixture attention

HYPOTHESIS: The unrestricted per-lag lookup is not load-bearing: three learned Gaussian delay components per head will retain at least 99% accuracy while reducing routing parameters from `n_head * (max_seq_len - 1)` to `8 * n_head`, because fixed-format routing should require only a few stationary delay modes.

INTENDED_EDIT: Replace each head’s dense relative-lag table with a learned three-component continuous delay mixture, retaining independent centers and widths per head and gauge-fixed mixture logits; leave token representation, value projection, MLP, checkpoints, and decoding unchanged.

EVIDENCE: Content-independent dual-head learned-lag routing reached 99.85%, showing stationary learned routes suffice, while reducing token-representation rank collapsed to 5.06%; this motivates preserving representation capacity while challenging the shared assumption that stationary routing needs an unrestricted logit for every lag.

<<<<<<< SEARCH
        # Each head learns a stationary causal routing preference. The omitted
        # final lag fixes the softmax-invariant common shift of each table.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )
=======
        # Each head learns a compact mixture of continuous delay kernels.
        # The final mixture logit is omitted as a softmax-invariant gauge.
        self.n_route = 3
        self.max_lag = max_seq_len - 1
        initial_centers = torch.linspace(0.2, 0.8, self.n_route)
        self.route_center = nn.Parameter(
            initial_centers.repeat(n_head, 1)
        )
        self.route_log_precision = nn.Parameter(
            torch.full(
                (n_head, self.n_route),
                math.log(4.0),
            )
        )
        self.route_mix = nn.Parameter(
            torch.zeros(n_head, self.n_route - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        positions = torch.arange(seqlen, device=x.device)
        lag = (positions[:, None] - positions[None, :]).clamp_min(0)
        lag_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=-1,
        )
        att = lag_bias[:, lag].unsqueeze(0)
=======
        positions = torch.arange(seqlen, device=x.device)
        lag = (positions[:, None] - positions[None, :]).clamp_min(0)
        lag_fraction = lag.to(dtype=x.dtype) / float(
            max(1, self.max_lag)
        )
        centers = self.route_center[:, None, None, :]
        precision = (
            self.route_log_precision.clamp(max=20.0)
            .exp()[:, None, None, :]
        )
        mixture_logits = torch.cat(
            (
                self.route_mix,
                self.route_mix.new_zeros(self.n_head, 1),
            ),
            dim=-1,
        )
        component_logits = (
            mixture_logits[:, None, None, :]
            - precision
            * (
                lag_fraction[None, :, :, None] - centers
            ).square()
        )
        att = torch.logsumexp(
            component_logits, dim=-1
        ).unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    relative_bias_gauge_parameters = [
        block.attn.relative_bias for block in model.blocks
    ]
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
=======
    route_mix_gauge_parameters = [
        block.attn.route_mix for block in model.blocks
    ]
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *route_mix_gauge_parameters,
    ]
>>>>>>> REPLACE