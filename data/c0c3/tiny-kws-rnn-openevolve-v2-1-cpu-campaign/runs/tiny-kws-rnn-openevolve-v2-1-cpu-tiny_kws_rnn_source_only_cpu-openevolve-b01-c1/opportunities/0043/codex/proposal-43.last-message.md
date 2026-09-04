MECHANISM: Progressive upper-skip adjacent-band folding

HYPOTHESIS: Folding normalized upper-skip bands 8–9 will retain at least 85% validation accuracy while reducing recurrent inference by exactly 3,286,080 MACs, because the lower GRU still receives all 19 folded spectral features.

INTENDED_EDIT: Reduce the upper GRU input width from 63 to 62 and replace its 14-feature acoustic skip with the first eight normalized bands plus averages of bands 8–9, 10–11, 12–13, 14–15, and 16–17 in both recurrent paths.

EVIDENCE: Four successive upper-skip adjacent-band folds preserved at least 85.40% accuracy and each saved exactly 3,286,080 recurrent MACs; the current model recovered to 85.52%, making the next identical localized fold the narrowest evidence-supported compression.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(63, 48, num_layers=1, batch_first=True)
=======
        self.upper_gru = nn.GRU(62, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :10],
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
=======
        lower_output = lower_output[:, 0, :]
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_acoustic = torch.cat(
            (
                normalized[..., :10],
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
>>>>>>> REPLACE