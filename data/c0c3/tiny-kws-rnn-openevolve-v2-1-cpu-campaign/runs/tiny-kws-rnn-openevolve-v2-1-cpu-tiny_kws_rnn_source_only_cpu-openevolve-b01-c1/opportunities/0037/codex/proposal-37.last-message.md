MECHANISM: Upper-skip adjacent-band folding

HYPOTHESIS: Folding normalized upper-skip bands 16–17 into one feature will preserve at least 85% validation accuracy because the lower GRU retains all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.

INTENDED_EDIT: Reduce the upper GRU input width from 67 to 66 and replace its 18-feature acoustic skip with the first 16 normalized bands plus the average of bands 16–17 in both recurrent paths.

EVIDENCE: Removing the folded highest-band feature only from the upper skip retained 85.77% accuracy and saved 3,286,080 recurrent MACs; prior whole-path experiments also showed adjacent-band folding preserves information better than deletion.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(67, 48, num_layers=1, batch_first=True)
=======
        self.upper_gru = nn.GRU(66, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_input = torch.cat((normalized[..., :18], lower_output), dim=-1)
=======
        upper_acoustic = torch.cat(
            (
                normalized[..., :16],
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_inputs = torch.cat((normalized[..., :18], lower_outputs), dim=-1)
=======
        upper_acoustic = torch.cat(
            (
                normalized[..., :16],
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
>>>>>>> REPLACE