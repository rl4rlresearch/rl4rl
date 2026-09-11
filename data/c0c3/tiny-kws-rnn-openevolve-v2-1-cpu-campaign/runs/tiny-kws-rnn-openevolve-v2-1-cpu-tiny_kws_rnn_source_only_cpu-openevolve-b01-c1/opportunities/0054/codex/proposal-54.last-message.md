MECHANISM: Learned low-band adjacent compression

HYPOTHESIS: A learned 2-to-1 projection of normalized bands 2–3 will retain at least 85% validation accuracy while reducing total inference by approximately 3.24M MACs.

INTENDED_EDIT: Add an averaging-initialized bias-free projection for bands 2–3, reduce the upper GRU input width from 57 to 56, and use the projection in both recurrent paths.

EVIDENCE: Fixed compression of bands 4–5 missed at 84.91%, while replacing it with a learned projection reached 86.50%; this tests the same information-preserving mechanism on the next adjacent low-band pair.

<<<<<<< SEARCH
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_band1013 = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.upper_band1013.weight, 0.25)
        self.upper_gru = nn.GRU(57, 48, num_layers=1, batch_first=True)
=======
        self.upper_band23 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band23.weight, 0.5)
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
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                self.upper_band1013(normalized[..., 10:14]),
            ),
            dim=-1,
        )
=======
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :2],
                self.upper_band23(normalized[..., 2:4]),
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                self.upper_band1013(normalized[..., 10:14]),
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
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                self.upper_band1013(normalized[..., 10:14]),
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
                normalized[..., :2],
                self.upper_band23(normalized[..., 2:4]),
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                self.upper_band1013(normalized[..., 10:14]),
            ),
            dim=-1,
        )
>>>>>>> REPLACE