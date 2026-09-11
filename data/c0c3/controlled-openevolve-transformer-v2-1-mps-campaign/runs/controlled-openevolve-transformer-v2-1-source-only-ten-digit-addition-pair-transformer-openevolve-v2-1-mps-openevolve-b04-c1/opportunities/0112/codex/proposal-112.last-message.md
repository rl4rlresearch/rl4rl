MECHANISM: Symmetric sparse-edge relative-bias pruning

HYPOTHESIS: Fixing both heads’ second-farthest-lag biases at zero will reduce the model from 845 to 843 parameters while retaining at least 99% accuracy, because those biases affect only two causal edges per head and the adjacent maximum-lag biases were removed with 99.84% accuracy.

INTENDED_EDIT: Shorten both heads’ relative-bias vectors by one coordinate and append two fixed zero-bias endpoints using the existing vectorized padding operation.

EVIDENCE: The 845-parameter design achieved 99.84% after pruning both maximum-lag biases. Single-head second-farthest pruning attempts timed out without negative accuracy evidence; symmetric pruning avoids their asymmetric reconstruction while testing the same sparse-edge hypothesis.

<<<<<<< SEARCH
        self.relative_bias_coordinates = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 2)
        )
=======
        self.relative_bias_coordinates = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias = F.pad(self.relative_bias_coordinates, (1, 1))
=======
        relative_bias = F.pad(self.relative_bias_coordinates, (1, 2))
>>>>>>> REPLACE