MECHANISM: Nonterminal-paired scalar MLP input

HYPOTHESIS: Sharing the fifth and sixth effective MLP input coefficients while keeping the seventh coefficient independent will reduce the model from 999 to 998 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Store six `fc1` coefficients, reuse the fifth for the sixth effective input channel, preserve the terminal coefficient, and consume constructor and initialization draws for the removed parameter.

EVIDENCE: Terminal `fc1` sharing narrowly missed at 98.95%, while changing the terminal coefficient toward its fixed reference collapsed to 26.89%; preserving that reference-sensitive terminal coefficient and moving the sharing constraint to the neighboring nonterminal pair isolates whether the failure was specific to the terminal geometry.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 1, d_ff, bias=False)

        # Preserve the RNG stream of the removed fc1 bias.
        fc1_bias_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(d_ff).uniform_(
            -fc1_bias_bound, fc1_bias_bound
        )
=======
        self.fc1 = nn.Linear(d_model - 2, d_ff, bias=False)
        self.fc1._removed_input_features = 1

        # Preserve the constructor RNG stream of the removed input weight.
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
                self.fc1.weight[:, :-1],
                self.fc1.weight[:, -2:-1],
                self.fc1.weight[:, -1:],
            ],
            dim=-1,
        )
        hidden = F.linear(x[..., :-1], fc1_weight)
        y = F.pad(self.fc2(F.gelu(hidden)), (0, 1))
        return self.drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                removed_input_features = getattr(
                    module, "_removed_input_features", 0
                )
                if removed_input_features:
                    discarded = module.weight.new_empty(
                        module.out_features, removed_input_features
                    )
                    nn.init.normal_(discarded, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE