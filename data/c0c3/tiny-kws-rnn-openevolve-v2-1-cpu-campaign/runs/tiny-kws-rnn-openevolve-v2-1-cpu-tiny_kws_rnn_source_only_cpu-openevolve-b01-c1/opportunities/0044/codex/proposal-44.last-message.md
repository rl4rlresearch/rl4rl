MECHANISM: Progressive upper-skip adjacent-band folding

HYPOTHESIS: Folding normalized upper-skip bands 6–7 will retain at least 85% validation accuracy because the lower GRU still receives all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.

INTENDED_EDIT: Reduce the upper GRU input width from 62 to 61 and replace its eight individual low-band skip features with six individual bands plus an average of bands 6–7 in both recurrent execution paths.

EVIDENCE: Five successive upper-skip reductions preserved at least 85% accuracy and each adjacent-band fold saved exactly 3,286,080 recurrent MACs; applying the same localized fold to the next pair directly tests the compression boundary.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(62, 48, num_layers=1, batch_first=True)
=======
        self.upper_gru = nn.GRU(61, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_acoustic = torch.cat(
            (
                normalized[..., :8],
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
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_acoustic = torch.cat(
            (
                normalized[..., :8],
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
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
>>>>>>> REPLACE