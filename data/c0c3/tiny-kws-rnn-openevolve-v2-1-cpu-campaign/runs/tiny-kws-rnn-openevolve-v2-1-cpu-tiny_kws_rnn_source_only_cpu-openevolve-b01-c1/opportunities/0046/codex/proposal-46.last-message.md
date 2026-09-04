MECHANISM: Frequency-aware upper-skip pruning

HYPOTHESIS: Removing the highest remaining upper-skip aggregate (bands 16–17) will retain at least 85% validation accuracy while saving exactly 3,286,080 recurrent MACs, because the lower GRU still receives all 19 features and prior failure on bands 4–5 suggests low-frequency skips are more important.

INTENDED_EDIT: Reduce the upper GRU input width from 61 to 60 and omit the normalized bands 16–17 average from both recurrent execution paths.

EVIDENCE: Deleting the highest-band feature only from the upper skip retained 85.77%, whereas folding low-band skip features 4–5 missed the requirement at 84.91%; this motivates preserving low bands and pruning the highest remaining redundant skip feature.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(61, 48, num_layers=1, batch_first=True)
=======
        self.upper_gru = nn.GRU(60, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_acoustic = torch.cat(
            (
                normalized[..., :6],
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
=======
        upper_acoustic = torch.cat(
            (
                normalized[..., :6],
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_acoustic = torch.cat(
            (
                normalized[..., :6],
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
=======
        upper_acoustic = torch.cat(
            (
                normalized[..., :6],
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
>>>>>>> REPLACE