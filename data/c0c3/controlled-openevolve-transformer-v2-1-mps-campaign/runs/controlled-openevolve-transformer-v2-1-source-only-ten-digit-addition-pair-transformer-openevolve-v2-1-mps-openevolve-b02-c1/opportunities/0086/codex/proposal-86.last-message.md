MECHANISM: Reference-skewed terminal MLP input interpolation

HYPOTHESIS: Reconstructing the seventh effective MLP input coefficient three-quarters of the way from the sixth coefficient toward the fixed zero reference will reduce the model from 999 to 998 parameters while exceeding 99% accuracy.

INTENDED_EDIT: Store six `fc1` input coefficients, derive the seventh as `0.75` times the sixth, and preserve the original constructor and initialization RNG streams.

EVIDENCE: Directly sharing the terminal `fc1` coefficients narrowly missed at 98.95%, while introducing a distinct quarter-step transition instead of equality previously raised head-1 relative-bias compression from failure to 99.97%.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 1, d_ff, bias=False)

        # Preserve the RNG stream of the removed fc1 bias.
        fc1_bias_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(d_ff).uniform_(
            -fc1_bias_bound, fc1_bias_bound
        )
=======
        self.fc1 = nn.Linear(d_model - 2, d_ff, bias=False)
        self.fc1._interpolated_input = True

        # Preserve the RNG stream of the original seven-input fc1.
        fc1_weight_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(d_ff).uniform_(
            -fc1_weight_bound, fc1_weight_bound
        )

        # Preserve the RNG stream of the removed fc1 bias.
        fc1_bias_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(d_ff).uniform_(
            -fc1_bias_bound, fc1_bias_bound
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.fc2(F.gelu(self.fc1(x[..., :-1]))), (0, 1))
        return self.drop(y)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = torch.cat(
            [
                self.fc1.weight,
                0.75 * self.fc1.weight[:, -1:],
            ],
            dim=1,
        )
        hidden = F.linear(x[..., :-1], fc1_weight)
        y = F.pad(self.fc2(F.gelu(hidden)), (0, 1))
        return self.drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "_interpolated_input", False):
                full_weight = module.weight.new_empty(
                    module.out_features, module.in_features + 1
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[:, :-1])
            elif getattr(module, "_residual_gauge", False):
                full_weight = module.weight.new_empty(
                    module.out_features + 1, module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[:-1] - full_weight[-1:])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE