MECHANISM: Absorb the second LayerNorm offset into full MLP hidden biases

HYPOTHESIS: Restoring all 12 `fc1` biases while removing the redundant eight-coordinate `ln2` offset will produce 1593 learned parameters and retain at least 99% accuracy.

INTENDED_EDIT: Disable only `ln2.bias` and restore the trailing `fc1` bias, leaving LayerNorm scaling, initialization streams, attention, and training unchanged.

EVIDENCE: The 1600-parameter design reached 99.91% with its missing hidden bias compensated by `ln2`’s offset, while removing another hidden bias fell to 98.57%; a full `fc1` bias supplies every unit’s threshold directly, making the preceding LayerNorm offset exactly absorbable into that bias.

<<<<<<< SEARCH
        # Construct the baseline layer first to preserve its initialization RNG
        # stream, then fix one hidden-unit bias coordinate at zero.
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        return self.drop(self.fc2(F.gelu(hidden)))
=======
        # A full hidden bias absorbs the affine offset of the preceding
        # LayerNorm, so that offset need not remain separately learned.
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model)
        # fc1.bias represents W @ ln2.bias plus the original hidden bias.
        self.ln2.bias = None
>>>>>>> REPLACE