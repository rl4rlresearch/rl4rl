MECHANISM: Cached quotient-gradient reconstruction with adjacent head-0 terminal-knot sharing

HYPOTHESIS: Sharing head 0’s final two learned relative-bias coefficients will produce a 980-parameter model with at least 99% accuracy, while reusing the virtual full-gradient reconstructions already computed during clipping and validating only at the final step will avoid the timeout that prevented the same compression from being evaluated.

INTENDED_EDIT: Remove one head-0 routing parameter, reconstruct the adjacent terminal distance from its final learned coefficient, cache exact clipped score/value quotient gradients for the optimizer step, and eliminate pre-final validation.

EVIDENCE: The 981-parameter design achieved 99.89% with five farthest head-0 distances fixed to zero. The prior adjacent-terminal-knot compression preserved that verified cutoff and failed only by timeout; subsequent runtime work did not address the duplicated quotient-gradient solves performed once during clipping and again during each optimizer step.

<<<<<<< SEARCH
        # Head 0 ties the five farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 5)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
=======
        # Head 0 shares its two terminal learned distances and ties the five
        # farthest distances. Head 1 ties the three farthest to its reference,
        # shares the next boundary quadruplet, and reconstructs the two
        # preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 6)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.rel_bias[0], (0, 5)),
=======
                torch.cat(
                    [
                        self.rel_bias[0],
                        self.rel_bias[0][-1:],
                        self.rel_bias[0].new_zeros(5),
                    ]
                ),
>>>>>>> REPLACE

<<<<<<< SEARCH
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)
=======
    def zero_grad(self, set_to_none: bool = True) -> None:
        self._cached_score_grads = None
        self._cached_attention_grads = None
        self.base.zero_grad(set_to_none=set_to_none)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Replace composite score gradients with those of the virtual
        # full-width query, key, and query-bias parameters.
        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
        ), state in zip(self.score_specs, self.score_states):
            full_grads = self._full_score_grads(
                query_param, key_tail_param, bias_param, state
            )
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())
=======
        # Replace composite score gradients with those of the virtual
        # full-width query, key, and query-bias parameters. Retain these
        # reconstructions so step() does not repeat the same solves.
        score_full_grads = []
        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
        ), state in zip(self.score_specs, self.score_states):
            full_grads = self._full_score_grads(
                query_param, key_tail_param, bias_param, state
            )
            score_full_grads.append(full_grads)
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Replace folded attention gradients with gradients of the virtual
        # value, LayerNorm, full projection, and shared-bias parameters.
        for (
            weight,
            _,
            proj_weight,
            folded_bias,
        ), state in zip(
            self.attention_specs, self.attention_states
        ):
            full_grads = self._full_attention_grads(
                weight, proj_weight, folded_bias, state
            )
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())

        total_norm = total_sq.sqrt()
        clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
=======
        # Replace folded attention gradients with gradients of the virtual
        # value, LayerNorm, full projection, and shared-bias parameters.
        attention_full_grads = []
        for (
            weight,
            _,
            proj_weight,
            folded_bias,
        ), state in zip(
            self.attention_specs, self.attention_states
        ):
            full_grads = self._full_attention_grads(
                weight, proj_weight, folded_bias, state
            )
            attention_full_grads.append(full_grads)
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())

        total_norm = total_sq.sqrt()
        clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        self._cached_score_grads = [
            tuple(
                grad * clip_coef.to(
                    device=grad.device, dtype=grad.dtype
                )
                for grad in full_grads
            )
            for full_grads in score_full_grads
        ]
        self._cached_attention_grads = [
            tuple(
                grad * clip_coef.to(
                    device=grad.device, dtype=grad.dtype
                )
                for grad in full_grads
            )
            for full_grads in attention_full_grads
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
        ), state in zip(self.score_specs, self.score_states):
            if (
                query_param.grad is None
                and key_tail_param.grad is None
                and bias_param.grad is None
            ):
                continue

            query_grad, key_grad, bias_grad = self._full_score_grads(
                query_param, key_tail_param, bias_param, state
            )
=======
        for score_index, ((
            query_param,
            key_tail_param,
            bias_param,
            _,
        ), state) in enumerate(zip(self.score_specs, self.score_states)):
            if (
                query_param.grad is None
                and key_tail_param.grad is None
                and bias_param.grad is None
            ):
                continue

            cached_score_grads = getattr(
                self, "_cached_score_grads", None
            )
            if cached_score_grads is None:
                query_grad, key_grad, bias_grad = self._full_score_grads(
                    query_param, key_tail_param, bias_param, state
                )
            else:
                query_grad, key_grad, bias_grad = cached_score_grads[
                    score_index
                ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            weight,
            _,
            proj_weight,
            folded_bias,
        ), state in zip(
            self.attention_specs, self.attention_states
        ):
            if (
                weight.grad is None
                and proj_weight.grad is None
                and folded_bias.grad is None
            ):
                continue

            (
                weight_grad,
                scale_grad,
                shift_grad,
                proj_weight_grad,
                full_bias_grad,
            ) = self._full_attention_grads(
                weight, proj_weight, folded_bias, state
            )
=======
        for attention_index, ((
            weight,
            _,
            proj_weight,
            folded_bias,
        ), state) in enumerate(zip(
            self.attention_specs, self.attention_states
        )):
            if (
                weight.grad is None
                and proj_weight.grad is None
                and folded_bias.grad is None
            ):
                continue

            cached_attention_grads = getattr(
                self, "_cached_attention_grads", None
            )
            if cached_attention_grads is None:
                (
                    weight_grad,
                    scale_grad,
                    shift_grad,
                    proj_weight_grad,
                    full_bias_grad,
                ) = self._full_attention_grads(
                    weight, proj_weight, folded_bias, state
                )
            else:
                (
                    weight_grad,
                    scale_grad,
                    shift_grad,
                    proj_weight_grad,
                    full_bias_grad,
                ) = cached_attention_grads[attention_index]
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE