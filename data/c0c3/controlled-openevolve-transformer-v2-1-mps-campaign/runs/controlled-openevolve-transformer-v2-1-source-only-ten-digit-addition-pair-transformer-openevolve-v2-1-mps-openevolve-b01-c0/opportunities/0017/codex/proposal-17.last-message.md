MECHANISM: Terminal LayerNorm-null MLP weight-column gauge

HYPOTHESIS: Centering one `fc2` output column will reduce the model to 1630 parameters while retaining at least 99% accuracy, because the removed input-dependent common offset is exactly canceled by final LayerNorm and all eight effective column coordinates remain represented through seven balanced contrasts.

INTENDED_EDIT: Replace the first `fc2` weight column with seven learned centered contrasts, retain the other eleven columns unchanged, and preserve the original initialization and RNG sequence.

EVIDENCE: Centering the terminal `fc2` bias retained 99.93% accuracy at 1631 parameters, showing that a balanced gauge immediately before final LayerNorm trains successfully; reducing only one weight-column gauge is the smallest extension of that proven mechanism.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.fc2.bias = None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        output_bias = torch.cat((self.fc2_bias, self.fc2_bias.new_zeros(1)))
        output_bias = output_bias - output_bias.mean()
        return self.drop(F.linear(hidden, self.fc2.weight, output_bias))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2_first_column = nn.Parameter(torch.empty(d_model - 1))
        self.fc2_rest = nn.Parameter(torch.empty(d_model, d_ff - 1))
        self.fc2.weight = None
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.fc2.bias = None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        final_coordinate = -self.fc2_first_column.sum().reshape(1)
        first_column = torch.cat((self.fc2_first_column, final_coordinate))
        weight = torch.cat((first_column.unsqueeze(1), self.fc2_rest), dim=1)
        output_bias = torch.cat((self.fc2_bias, self.fc2_bias.new_zeros(1)))
        output_bias = output_bias - output_bias.mean()
        return self.drop(F.linear(hidden, weight, output_bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, MLP):
            full_weight = module.fc2_first_column.new_empty(
                module.fc2.out_features, module.fc2.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            first_column = full_weight[:, 0] - full_weight[:, 0].mean()
            with torch.no_grad():
                module.fc2_first_column.copy_(first_column[:-1])
                module.fc2_rest.copy_(full_weight[:, 1:])
>>>>>>> REPLACE