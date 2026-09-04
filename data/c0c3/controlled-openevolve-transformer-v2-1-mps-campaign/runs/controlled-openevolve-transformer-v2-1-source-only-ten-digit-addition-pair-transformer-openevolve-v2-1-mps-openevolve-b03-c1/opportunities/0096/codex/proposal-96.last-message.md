MECHANISM: Cross-head twenty-third-distance bias tying

HYPOTHESIS: Sharing both heads’ twenty-third-farthest attention bias will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy, because all nine adjacent distance ties through the twenty-second-farthest distance succeeded.

INTENDED_EDIT: Shorten the head-specific relative-bias core by one distance, add one shared learned scalar for the removed twenty-third-farthest distance, and insert it immediately before the existing twenty-second-distance scalar.

EVIDENCE: The twenty-second-distance tie achieved 99.91% accuracy with 1,107 parameters; prior twenty-third-distance submissions yielded no negative accuracy evidence because they reproduced the unchanged implementation or could not be verified.

<<<<<<< SEARCH
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 23))
        self.relative_bias_core_twenty_second = nn.Parameter(torch.zeros(1))
=======
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 24))
        self.relative_bias_core_twenty_third = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_twenty_second = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_core_twenty_second = (
            self.relative_bias_core_twenty_second.expand(
                self.n_head
            ).unsqueeze(1)
        )
=======
        relative_bias_core_twenty_third = (
            self.relative_bias_core_twenty_third.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_twenty_second = (
            self.relative_bias_core_twenty_second.expand(
                self.n_head
            ).unsqueeze(1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias,
                relative_bias_core_twenty_second,
                relative_bias_core_twenty_first,
=======
                self.relative_bias,
                relative_bias_core_twenty_third,
                relative_bias_core_twenty_second,
                relative_bias_core_twenty_first,
>>>>>>> REPLACE