MECHANISM: Progressive upper-skip adjacent-band folding

HYPOTHESIS: Folding normalized upper-skip bands 10–11 into one feature will retain at least 85% validation accuracy while reducing recurrent inference by exactly 3,286,080 MACs, because the lower GRU still receives all 19 folded spectral features.

INTENDED_EDIT: Reduce the upper GRU input width from 64 to 63 and replace its 15-feature acoustic skip with the first 10 normalized bands plus averages of bands 10–11, 12–13, 14–15, and 16–17 in both recurrent paths.

EVIDENCE: Three successive upper-skip folds preserved at least 85.40% accuracy and each saved exactly 3,286,080 recurrent MACs; applying the same localized, information-preserving reduction to the next adjacent pair is the narrowest supported compression.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(64, 48, num_layers=1, batch_first=True)
=======
        self.upper_gru = nn.GRU(63, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :12],
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
=======
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
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                normalized[..., :12],
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
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
                normalized[..., :10],
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
>>>>>>> REPLACE