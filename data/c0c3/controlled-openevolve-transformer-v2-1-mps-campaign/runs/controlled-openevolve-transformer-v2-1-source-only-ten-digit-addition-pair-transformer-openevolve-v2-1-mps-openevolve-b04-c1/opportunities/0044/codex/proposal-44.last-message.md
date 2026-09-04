MECHANISM: Ambient-state AdamW for a quotiented MLP channel

HYPOTHESIS: Quotienting the penultimate `fc2` column while updating it through eight-dimensional AdamW moments will retain at least 99% accuracy with 1601 parameters, because it preserves the successful 1602-parameter model’s dense-column optimization trajectory modulo the exact final-LayerNorm shift symmetry.

INTENDED_EDIT: Represent the penultimate `fc2` column with seven Householder coordinates and optimize those coordinates using projected updates from ambient eight-dimensional AdamW state.

EVIDENCE: The same penultimate-column quotient fell to 35.4% under coordinate-wise AdamW, while the dense penultimate column in the 1602-parameter design reached 99.99%; this directly tests whether Adam’s lack of rotation invariance, rather than functional capacity, caused that failure.

<<<<<<< SEARCH
        self.first_coordinates = nn.Parameter(torch.empty(2, out_features - 1))
        self.rest_weight = nn.Parameter(torch.empty(out_features, in_features - 3))
        self.last_coordinates = nn.Parameter(torch.empty(out_features - 1))
=======
        self.first_coordinates = nn.Parameter(torch.empty(2, out_features - 1))
        self.rest_weight = nn.Parameter(torch.empty(out_features, in_features - 4))
        self.middle_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.last_coordinates = nn.Parameter(torch.empty(out_features - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _set_rest(self, conceptual_rest: torch.Tensor) -> None:
        with torch.no_grad():
            self.rest_weight.copy_(conceptual_rest[:, :-1])
            transformed_last = self._householder(conceptual_rest[:, -1])
            self.last_coordinates.copy_(transformed_last[1:])
=======
    def _set_rest(self, conceptual_rest: torch.Tensor) -> None:
        with torch.no_grad():
            self.rest_weight.copy_(conceptual_rest[:, :-2])
            transformed_middle = self._householder(conceptual_rest[:, -2])
            self.middle_coordinates.copy_(transformed_middle[1:])
            transformed_last = self._householder(conceptual_rest[:, -1])
            self.last_coordinates.copy_(transformed_last[1:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        last = F.pad(self.last_coordinates, (1, 0))
        last = self._householder(last)
        weight = torch.cat(
            (first.transpose(0, 1), self.rest_weight, last.unsqueeze(1)), dim=1
        )
=======
        middle = F.pad(self.middle_coordinates, (1, 0))
        middle = self._householder(middle)
        last = F.pad(self.last_coordinates, (1, 0))
        last = self._householder(last)
        weight = torch.cat(
            (
                first.transpose(0, 1),
                self.rest_weight,
                middle.unsqueeze(1),
                last.unsqueeze(1),
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.sample_pairs()
        return encode_batch(a, b)


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float) -> float:
=======
    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.sample_pairs()
        return encode_batch(a, b)


class AmbientGaugeAdamW:
    """AdamW in materialized space, projected back to gauge coordinates."""

    def __init__(
        self,
        entries: List[Tuple[torch.nn.Parameter, torch.Tensor, float]],
        lr: float,
        weight_decay: float,
    ):
        self.entries = entries
        self.param_groups = [{"lr": lr}]
        self.weight_decay = weight_decay
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.states = [
            {
                "step": 0,
                "exp_avg": torch.zeros_like(reflector),
                "exp_avg_sq": torch.zeros_like(reflector),
            }
            for _, reflector, _ in entries
        ]

    @staticmethod
    def _householder(
        value: torch.Tensor, reflector: torch.Tensor, norm_sq: float
    ) -> torch.Tensor:
        projection = 2.0 * torch.dot(reflector, value) / norm_sq
        return value - reflector * projection

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter, _, _ in self.entries:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        lr = self.param_groups[0]["lr"]
        for (parameter, reflector, norm_sq), state in zip(
            self.entries, self.states
        ):
            if parameter.grad is None:
                continue

            padded = torch.cat((parameter.new_zeros(1), parameter))
            ambient = self._householder(padded, reflector, norm_sq)
            grad_padded = torch.cat(
                (parameter.grad.new_zeros(1), parameter.grad)
            )
            ambient_grad = self._householder(
                grad_padded, reflector, norm_sq
            )

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                ambient_grad, alpha=1.0 - self.beta1
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                ambient_grad, ambient_grad, value=1.0 - self.beta2
            )

            ambient.mul_(1.0 - lr * self.weight_decay)
            bias_correction1 = 1.0 - self.beta1 ** state["step"]
            bias_correction2 = 1.0 - self.beta2 ** state["step"]
            denom = (
                exp_avg_sq.sqrt()
                .div(math.sqrt(bias_correction2))
                .add(self.eps)
            )
            ambient.addcdiv_(
                exp_avg, denom, value=-lr / bias_correction1
            )

            projected = self._householder(
                ambient, reflector, norm_sq
            )
            parameter.copy_(projected[1:])


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float) -> float:
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    gauge_entries = [
        (
            block.mlp.fc2.middle_coordinates,
            block.mlp.fc2.reflector,
            block.mlp.fc2.reflector_norm_sq,
        )
        for block in model.blocks
    ]
    gauge_ids = {id(parameter) for parameter, _, _ in gauge_entries}
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if id(parameter) not in gauge_ids],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizer = AmbientGaugeAdamW(
        gauge_entries,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
=======
        for active_optimizer in (optimizer, gauge_optimizer):
            for pg in active_optimizer.param_groups:
                pg["lr"] = lr_now
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        gauge_optimizer.step()
>>>>>>> REPLACE