MECHANISM: Frequency-aware upper-skip pruning

HYPOTHESIS: Omitting the highest remaining upper-skip aggregate, bands 12–13, will retain at least 85% validation accuracy while reducing recurrent inference by exactly 3,286,080 MACs.

INTENDED_EDIT: Reduce the upper GRU input width from 59 to 58 and remove the normalized bands 12–13 average from both recurrent execution paths.

EVIDENCE: Removing the bands 14–15 aggregate improved accuracy to 85.89% while saving exactly 3,286,080 recurrent MACs; continuing high-to-low pruning is better supported than folding bands 4–5, which failed at 84.91%.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(59, 48, num_layers=1, batch_first=True)
=======
        self.upper_gru = nn.GRU(58, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
=======
        upper_acoustic = torch.cat(
            (
                normalized[..., :6],
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
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
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
>>>>>>> REPLACE