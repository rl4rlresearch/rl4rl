MECHANISM: Asymmetric next-edge relative-bias pruning

HYPOTHESIS: Fixing the unbiased head’s second-farthest-lag bias at zero will reduce the model from 845 to 844 parameters while retaining at least 99% accuracy, because the successful maximum-lag pruning achieved 99.84% and this removes only one additional sparsely used positional-bias coordinate.

INTENDED_EDIT: Flatten the learned relative-bias coordinates, omit the unbiased head’s final learned coordinate, and reconstruct that second-farthest-lag bias as zero before attention.

EVIDENCE: Fixing both heads’ maximum-lag biases at zero retained 99.84% accuracy at 845 parameters, indicating substantial margin for testing the immediately adjacent sparse attention edge one head at a time.

<<<<<<< SEARCH
        # A constant shift of every lag bias cancels in the softmax, so lag
        # zero is the fixed reference coordinate.
        self.relative_bias_coordinates = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 2)
        )
=======
        # A constant shift of every lag bias cancels in the softmax, so lag
        # zero is the fixed reference coordinate. The final coordinate of the
        # unbiased head is also fixed, pruning its second-farthest-lag bias.
        self.relative_bias_width = max_seq_len - 2
        self.relative_bias_coordinates = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_width - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias = F.pad(self.relative_bias_coordinates, (1, 1))
        att = att + relative_bias[:, lags].unsqueeze(0)
=======
        relative_bias_coordinates = F.pad(
            self.relative_bias_coordinates, (0, 1)
        ).view(self.n_head, self.relative_bias_width)
        relative_bias = F.pad(relative_bias_coordinates, (1, 1))
        att = att + relative_bias[:, lags].unsqueeze(0)
>>>>>>> REPLACE