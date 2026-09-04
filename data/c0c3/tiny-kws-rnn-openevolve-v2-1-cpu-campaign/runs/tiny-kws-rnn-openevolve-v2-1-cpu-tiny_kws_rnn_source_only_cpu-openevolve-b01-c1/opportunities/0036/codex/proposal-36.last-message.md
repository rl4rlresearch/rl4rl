MECHANISM: Redundant upper-stage spectral-skip pruning

HYPOTHESIS: Removing only the folded highest-band feature from the upper GRU’s raw acoustic skip will preserve at least 85% validation accuracy because the lower GRU still receives all 19 features, while reducing recurrent inference by exactly 3,286,080 MACs.

INTENDED_EDIT: Reduce the upper GRU input from 68 to 67 features and pass it the first 18 normalized acoustic features plus the complete 49-unit lower recurrent output in both execution paths.

EVIDENCE: The 49+48 hierarchy achieved 86.01% accuracy, while removing a spectral feature from the entire recurrent path failed; retaining all spectral information in the more load-bearing lower stage while pruning only its redundant direct copy is a narrower structural reduction.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(68, 48, num_layers=1, batch_first=True)
=======
        self.upper_gru = nn.GRU(67, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_input = torch.cat((normalized, lower_output), dim=-1)
=======
        upper_input = torch.cat((normalized[..., :18], lower_output), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_inputs = torch.cat((normalized, lower_outputs), dim=-1)
=======
        upper_inputs = torch.cat((normalized[..., :18], lower_outputs), dim=-1)
>>>>>>> REPLACE