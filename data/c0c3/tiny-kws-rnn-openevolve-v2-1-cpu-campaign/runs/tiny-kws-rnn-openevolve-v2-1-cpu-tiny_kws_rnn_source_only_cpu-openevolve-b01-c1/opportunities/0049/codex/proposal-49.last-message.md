MECHANISM: Learned adjacent-band compression

HYPOTHESIS: Replacing normalized bands 4–5 with a learned 2-to-1 projection will retain at least 85% validation accuracy by learning the discriminative direction that fixed averaging lost, while reducing total inference by approximately 3.24M MACs.

INTENDED_EDIT: Reduce the upper GRU input width from 59 to 58 and compress bands 4–5 through a bias-free linear projection initialized to their average in both recurrent paths.

EVIDENCE: Fixed averaging of bands 4–5 narrowly missed the requirement at 84.91%, while retaining both features achieved 85.89%; a learned scalar projection tests whether the failure came from the fixed equal-weight aggregation rather than the one-dimensional bottleneck.

<<<<<<< SEARCH
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_gru = nn.GRU(59, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
=======
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_gru = nn.GRU(58, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :6],
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
=======
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                normalized[..., :6],
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
=======
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
>>>>>>> REPLACE