MECHANISM: Frequency-aware upper-skip pruning

HYPOTHESIS: Omitting the highest remaining upper-skip aggregate, bands 14–15, will retain at least 85% validation accuracy while reducing recurrent inference by exactly 3,286,080 MACs.

INTENDED_EDIT: Reduce the upper GRU input width from 60 to 59 and remove the normalized bands 14–15 average from both recurrent execution paths.

EVIDENCE: Removing the bands 16–17 upper-skip aggregate achieved 85.52% accuracy and saved exactly 3,286,080 recurrent MACs, while folding lower-frequency bands 4–5 failed; this supports continuing frequency-aware pruning from the high end.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(60, 48, num_layers=1, batch_first=True)
=======
        self.upper_gru = nn.GRU(59, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
=======
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
=======
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
>>>>>>> REPLACE