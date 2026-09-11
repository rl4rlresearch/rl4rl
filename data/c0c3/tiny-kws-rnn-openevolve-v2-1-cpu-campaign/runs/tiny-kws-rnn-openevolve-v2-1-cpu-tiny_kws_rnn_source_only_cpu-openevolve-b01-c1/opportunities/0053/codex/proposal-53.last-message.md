MECHANISM: Learned mid-band skip fusion

HYPOTHESIS: Replacing the separate bands 6–7 and 8–9 upper-skip averages with one learned four-band projection will retain at least 85% validation accuracy while reducing total inference by approximately 3.19M MACs.

INTENDED_EDIT: Add a bias-free 4-to-1 projection initialized to averaging, use it to fuse normalized bands 6–9 in both recurrent paths, and reduce the upper GRU input width from 57 to 56.

EVIDENCE: The learned four-band fusion of bands 10–13 retained 85.15% accuracy while saving approximately 3.19M MACs, and the separate prior folds of bands 6–7 and 8–9 each met the accuracy requirement; this directly tests whether those two already-compressed features can share one learned direction.

<<<<<<< SEARCH
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_band1013 = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.upper_band1013.weight, 0.25)
        self.upper_gru = nn.GRU(57, 48, num_layers=1, batch_first=True)
=======
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_band69 = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.upper_band69.weight, 0.25)
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
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                self.upper_band1013(normalized[..., 10:14]),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
=======
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
                self.upper_band69(normalized[..., 6:10]),
                self.upper_band1013(normalized[..., 10:14]),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
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
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                self.upper_band1013(normalized[..., 10:14]),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
=======
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
                self.upper_band69(normalized[..., 6:10]),
                self.upper_band1013(normalized[..., 10:14]),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
>>>>>>> REPLACE