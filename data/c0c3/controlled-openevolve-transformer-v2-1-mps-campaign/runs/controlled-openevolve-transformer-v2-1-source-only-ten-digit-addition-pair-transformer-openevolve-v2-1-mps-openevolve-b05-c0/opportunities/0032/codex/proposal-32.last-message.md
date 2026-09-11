MECHANISM: First-coordinate value/output-bias quotient

HYPOTHESIS: Fixing value-bias index 0 while retaining the successful fixed indices 6–7 will reduce the model from 1627 to 1626 parameters and maintain at least 99% accuracy, testing whether prior failures were coordinate-specific.

INTENDED_EDIT: Store five value-bias coordinates and reconstruct the eight-dimensional bias with zeros at indices 0, 6, and 7.

EVIDENCE: Fixing indices 6–7 achieved 99.92% at 1627 parameters, while additional removals at indices 3, 4, and 5 produced substantially different accuracies; this coordinate sensitivity makes the untested index 0 the smallest informative next reduction.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model - 2))
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v + F.pad(self.v_bias, (0, 2))
=======
        v = v + F.pad(self.v_bias, (1, 2))
>>>>>>> REPLACE