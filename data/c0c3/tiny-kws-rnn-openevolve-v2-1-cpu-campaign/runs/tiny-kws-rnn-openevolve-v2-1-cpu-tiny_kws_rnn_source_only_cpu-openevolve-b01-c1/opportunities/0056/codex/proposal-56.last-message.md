MECHANISM: Learned low-band subspace compression

HYPOTHESIS: Compressing normalized bands 0–3 from four features to a learned three-dimensional subspace will retain at least 85% validation accuracy while reducing total inference by approximately 3.01M MACs.

INTENDED_EDIT: Add a bias-free 4-to-3 projection initialized to preserve bands 0–1 and average bands 2–3, use it in both recurrent paths, and reduce the upper GRU input width from 57 to 56.

EVIDENCE: The learned bands 2–3 scalar projection missed the threshold by only 0.09 percentage points, while retaining four low-band dimensions achieved 85.15%; a three-dimensional learned projection starts from that tested compression but can recover discriminative mixtures across all four bands.

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
                torch.tensor(
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
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
=======
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                self.upper_band03(normalized[..., :4]),
                self.upper_band45(normalized[..., 4:6]),
>>>>>>> REPLACE

<<<<<<< SEARCH
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
=======
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                self.upper_band03(normalized[..., :4]),
                self.upper_band45(normalized[..., 4:6]),
>>>>>>> REPLACE