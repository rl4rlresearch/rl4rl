MECHANISM: Efficient symmetry-aligned MLP output sharing

HYPOTHESIS: Directly grouping MLP output coordinates 2–3 will reduce the model from 984 to 983 parameters and retain at least 99% accuracy while avoiding the prior implementation’s training timeout.

INTENDED_EDIT: Store six fc2 coefficients, reconstruct coordinates 2–3 from one shared coefficient, project the original eight-coordinate initialization onto the tie, and reuse the existing quotient optimizer without additional training-time machinery.

EVIDENCE: Coordinates 2–3 already share their MLP input coefficient and final-LayerNorm scale in the verified 99.75%-accurate model; the previous output-sharing attempt produced no accuracy result because training timed out, while every other tested 983-parameter constraint reduced accuracy below 99%.

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.fc2._residual_gauge = True

        # Preserve the RNG stream of the original full-width Linear constructor.
        bound = 1.0 / math.sqrt(d_ff)
        torch.empty(d_ff).uniform_(-bound, bound)
=======
        self.fc2 = nn.Linear(d_ff, d_model - 2, bias=False)
        self.fc2._residual_gauge = True
        self.fc2._tied_output_pair = True

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
        fc2_weight = torch.cat(
            [
                self.fc2.weight[:3],
                self.fc2.weight[2:3],
                self.fc2.weight[3:],
            ],
            dim=0,
        )
        y = F.pad(F.linear(F.gelu(hidden), fc2_weight), (0, 1))
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
                removed_outputs = (
                    2 if getattr(module, "_tied_output_pair", False) else 1
                )
                full_weight = module.weight.new_empty(
                    module.out_features + removed_outputs,
                    module.in_features,
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                effective_weight = full_weight[:-1] - full_weight[-1:]
                if getattr(module, "_tied_output_pair", False):
                    effective_weight = torch.cat(
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
                    module.weight.copy_(effective_weight)
>>>>>>> REPLACE