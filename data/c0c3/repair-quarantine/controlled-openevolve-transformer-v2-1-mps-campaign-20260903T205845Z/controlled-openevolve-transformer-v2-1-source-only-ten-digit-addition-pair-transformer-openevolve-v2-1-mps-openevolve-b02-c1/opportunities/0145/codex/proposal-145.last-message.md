MECHANISM: Symmetry-aligned MLP output sharing

HYPOTHESIS: Sharing MLP residual-output coefficients for coordinates 2 and 3 will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because these coordinates already share both their MLP input coefficient and final-LayerNorm scale, while the failed 983-parameter MLP experiment altered the more sensitive hidden activation.

INTENDED_EDIT: Store one shared fc2 coefficient for residual coordinates 2–3, reconstruct both outputs from it, and initialize it by projecting the original full-width initialization onto that equality constraint.

EVIDENCE: The verified 984-parameter model reached 99.75% after successfully sharing final-LayerNorm scales for coordinates 2–3. In contrast, sharing coordinate 6’s MLP input coefficient reduced accuracy to 98.24%, motivating an output-side constraint on the already aligned 2–3 coordinate pair that leaves MLP gating unchanged.

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.fc2._residual_gauge = True

        # Preserve the RNG stream of the original full-width Linear constructor.
        bound = 1.0 / math.sqrt(d_ff)
        torch.empty(d_ff).uniform_(-bound, bound)
=======
        self.fc2 = nn.Linear(d_ff, d_model - 2, bias=False)
        self.fc2._paired_residual_gauge = True

        # Preserve the RNG stream of the original full-width Linear constructor.
        bound = 1.0 / math.sqrt(d_ff)
        torch.empty(2 * d_ff).uniform_(-bound, bound)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = F.linear(x[..., :-1], fc1_weight)
        y = F.pad(self.fc2(F.gelu(hidden)), (0, 1))
        return self.drop(y)
=======
        hidden = F.linear(x[..., :-1], fc1_weight)
        reduced_y = self.fc2(F.gelu(hidden))
        paired_y = torch.cat(
            [
                reduced_y[..., :3],
                reduced_y[..., 2:3],
                reduced_y[..., 3:],
            ],
            dim=-1,
        )
        y = F.pad(paired_y, (0, 1))
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
=======
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "_paired_residual_gauge", False):
                full_weight = module.weight.new_empty(
                    module.out_features + 2, module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                effective_weight = full_weight[:-1] - full_weight[-1:]
                paired_weight = torch.cat(
                    [
                        effective_weight[:2],
                        effective_weight[2:4].mean(
                            dim=0, keepdim=True
                        ),
                        effective_weight[4:],
                    ],
                    dim=0,
                )
                with torch.no_grad():
                    module.weight.copy_(paired_weight)
            elif getattr(module, "_residual_gauge", False):
                full_weight = module.weight.new_empty(
                    module.out_features + 1, module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[:-1] - full_weight[-1:])
            else:
>>>>>>> REPLACE