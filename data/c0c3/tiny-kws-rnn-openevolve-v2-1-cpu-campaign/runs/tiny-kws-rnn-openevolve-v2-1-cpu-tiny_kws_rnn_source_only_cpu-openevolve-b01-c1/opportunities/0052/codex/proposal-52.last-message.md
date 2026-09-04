MECHANISM: Learned high-band skip fusion

HYPOTHESIS: Replacing the separate bands 10–11 and 12–13 upper-skip averages with one learned four-band projection will retain at least 85% validation accuracy while reducing total inference by approximately 3.19M MACs.

INTENDED_EDIT: Reduce the upper GRU input width from 58 to 57 and fuse normalized bands 10–13 through a bias-free learned projection initialized to their average in both recurrent paths.

EVIDENCE: Removing bands 12–13 entirely failed at 84.42%, but learned 2-to-1 compression of bands 4–5 recovered from a fixed-average failure to 86.50%; this tests whether a learned shared high-band direction preserves information that outright pruning lost.

<<<<<<< SEARCH
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_gru = nn.GRU(58, 48, num_layers=1, batch_first=True)
=======
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_band1013 = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.upper_band1013.weight, 0.25)
        self.upper_gru = nn.GRU(57, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
=======
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                self.upper_band1013(normalized[..., 10:14]),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
=======
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                self.upper_band1013(normalized[..., 10:14]),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
>>>>>>> REPLACE