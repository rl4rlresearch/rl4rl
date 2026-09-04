MECHANISM: Learned low-band subspace compression

HYPOTHESIS: Replacing four individual upper-skip bands 0–3 with a learned 4-to-3 projection will retain at least 85% validation accuracy while saving approximately 3.01M total MACs; unlike the narrowly failing bands 2–3 scalar projection, the unrestricted three-dimensional subspace can learn mixtures across all four low bands.

INTENDED_EDIT: Add an averaging-initialized bias-free 4-to-3 projection for normalized bands 0–3, use it in both recurrent paths, and reduce the upper GRU input width from 57 to 56.

EVIDENCE: The constrained learned projection of bands 2–3 missed the threshold by only 0.09 percentage points at 84.91%, while the current 57-input model achieved 85.15%; this projection initially reproduces that constrained representation but adds cross-band degrees of freedom at modest MAC cost.

<<<<<<< SEARCH
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_band1013 = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.upper_band1013.weight, 0.25)
        self.upper_gru = nn.GRU(57, 48, num_layers=1, batch_first=True)
=======
        self.upper_band03 = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.upper_band03.weight.copy_(
                self.upper_band03.weight.new_tensor(
                    (
                        (1.0, 0.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (0.0, 0.0, 0.5, 0.5),
                    )
                )
            )
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_band1013 = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.upper_band1013.weight, 0.25)
        self.upper_gru = nn.GRU(56, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
=======
                self.upper_band03(normalized[..., :4]),
                self.upper_band45(normalized[..., 4:6]),
>>>>>>> REPLACE

<<<<<<< SEARCH
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
=======
                self.upper_band03(normalized[..., :4]),
                self.upper_band45(normalized[..., 4:6]),
>>>>>>> REPLACE