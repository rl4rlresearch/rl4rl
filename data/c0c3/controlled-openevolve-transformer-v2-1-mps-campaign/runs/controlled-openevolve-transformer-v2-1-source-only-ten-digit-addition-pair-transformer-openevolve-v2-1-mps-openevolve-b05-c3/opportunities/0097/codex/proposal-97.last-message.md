MECHANISM: Five-way terminal lag-logit tying

HYPOTHESIS: Extending the verified four-coordinate terminal lag tie to five coordinates will reduce the model from 1,253 to 1,252 learned parameters while retaining at least 99% accuracy, because the remaining shared lag kernel preserves sufficient routing capacity.

INTENDED_EDIT: Constrain the final five cyclic-kernel logits to the gauge-fixed value, leaving the learned attention, optimizer, checkpoints, and decoding interface unchanged.

EVIDENCE: The four-way terminal tie achieved 99.91% accuracy with 1,253 parameters, and each preceding one-coordinate extension also exceeded 99%, motivating the next isolated reduction.

<<<<<<< SEARCH
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the four terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 4)
        )
=======
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the five terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 5)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(4),
            )
        )
=======
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(5),
            )
        )
>>>>>>> REPLACE