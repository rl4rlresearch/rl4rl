MECHANISM: Terminal-paired scalar MLP output

HYPOTHESIS: Sharing the final two learned MLP output contrasts will reduce the model from 999 to 998 parameters while retaining at least 99% accuracy, because terminal residual-channel sharing has succeeded in both final LayerNorm scale and bias parameters, and removing the scalar MLP offset retained 99.98% accuracy.

INTENDED_EDIT: Store six instead of seven `fc2` output contrasts, reconstruct the seventh by reusing the sixth, and preserve the existing constructor and initialization RNG streams.

EVIDENCE: Sharing the terminal final-LayerNorm bias pair achieved 99.98%, and fixing the width-one MLP bias at zero preserved 99.98%; together these results motivate testing one isolated terminal constraint on the remaining scalar MLP output vector without modifying sensitive attention pathways.

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.fc2._residual_gauge = True

        # Preserve the RNG stream of the original full-width Linear constructor.
        bound = 1.0 / math.sqrt(d_ff)
        torch.empty(d_ff).uniform_(-bound, bound)
=======
        self.fc2 = nn.Linear(d_ff, d_model - 2, bias=False)
        self.fc2._residual_gauge = True
        self.fc2._shared_terminal_output = True

        bound = 1.0 / math.sqrt(d_ff)

        # Preserve the removed gauge-fixed output row's constructor draw.
        torch.empty(d_ff).uniform_(-bound, bound)

        # Preserve the RNG stream of the original full-width Linear constructor.
        torch.empty(d_ff).uniform_(-bound, bound)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.fc2(F.gelu(self.fc1(x[..., :-1]))), (0, 1))
        return self.drop(y)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(F.gelu(self.fc1(x[..., :-1])))
        y = torch.cat([y, y[..., -1:]], dim=-1)
        y = F.pad(y, (0, 1))
        return self.drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
            if getattr(module, "_residual_gauge", False):
                full_weight = module.weight.new_empty(
                    module.out_features + 1, module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[:-1] - full_weight[-1:])
=======
            if getattr(module, "_residual_gauge", False):
                extra_rows = (
                    2
                    if getattr(module, "_shared_terminal_output", False)
                    else 1
                )
                full_weight = module.weight.new_empty(
                    module.out_features + extra_rows,
                    module.in_features,
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(
                        full_weight[: module.out_features]
                        - full_weight[-1:]
                    )
>>>>>>> REPLACE