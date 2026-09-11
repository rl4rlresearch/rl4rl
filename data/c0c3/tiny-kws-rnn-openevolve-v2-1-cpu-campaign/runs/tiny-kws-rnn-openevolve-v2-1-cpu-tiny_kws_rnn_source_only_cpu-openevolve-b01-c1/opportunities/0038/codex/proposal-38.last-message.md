MECHANISM: Progressive upper-skip adjacent-band folding

HYPOTHESIS: Folding normalized upper-skip bands 14–15 into one feature will preserve at least 85% validation accuracy because the lower GRU retains all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.

INTENDED_EDIT: Reduce the upper GRU input width from 66 to 65 and replace its 17-feature acoustic skip with the first 14 normalized bands plus averages of bands 14–15 and 16–17 in both recurrent paths.

EVIDENCE: Folding upper-skip bands 16–17 retained 85.77% accuracy and saved 3,286,080 recurrent MACs; applying the same information-preserving reduction to the next adjacent pair is the narrowest supported structural compression.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(66, 48, num_layers=1, batch_first=True)
=======
        self.upper_gru = nn.GRU(65, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :16],
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
=======
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :14],
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
                normalized[..., :16],
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
                normalized[..., :14],
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
>>>>>>> REPLACE