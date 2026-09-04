MECHANISM: Sparse tail-weight exponential averaging

HYPOTHESIS: Averaging only three late cosine-schedule checkpoints will exceed 9,322 correct predictions by reducing final optimizer noise without the timeout caused by updating an EMA after every step.

INTENDED_EDIT: Replace AdamW with a compatible subclass that samples weights at 80%, 90%, and 100% of training, forms a final-biased 25/25/50 average, and installs it after the exact final optimizer step.

EVIDENCE: The unchanged hard-maximum model reliably reached 9,320 correct, while the per-step ramped EMA experiment timed out; sparse checkpoint averaging tests the same promising stabilization mechanism with only three whole-model updates.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
=======
class SparseTailAveragedAdamW(torch.optim.AdamW):
    def __init__(
        self,
        parameters,
        total_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(parameters, **kwargs)
        self.completed_steps = 0
        self.total_steps = total_steps
        self.average_steps = {
            max(1, int(round(total_steps * fraction)))
            for fraction in (0.80, 0.90, 1.00)
        }
        self.tail_average = None

    def step(self, closure=None):
        loss = super().step(closure)
        self.completed_steps += 1

        if self.completed_steps in self.average_steps:
            with torch.no_grad():
                parameters = [
                    parameter
                    for group in self.param_groups
                    for parameter in group["params"]
                ]
                if self.tail_average is None:
                    self.tail_average = [
                        parameter.detach().clone() for parameter in parameters
                    ]
                else:
                    for average, parameter in zip(
                        self.tail_average, parameters
                    ):
                        average.lerp_(parameter.detach(), 0.5)

                if self.completed_steps == self.total_steps:
                    for parameter, average in zip(
                        parameters, self.tail_average
                    ):
                        parameter.copy_(average)

        return loss


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    return SparseTailAveragedAdamW(
        model.parameters(),
        total_steps=total_steps,
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE