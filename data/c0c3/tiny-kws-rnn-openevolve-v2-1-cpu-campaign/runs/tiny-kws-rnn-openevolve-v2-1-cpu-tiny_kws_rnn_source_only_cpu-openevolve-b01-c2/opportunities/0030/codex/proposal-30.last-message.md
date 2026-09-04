MECHANISM: Redundant affine-parameter pruning

HYPOTHESIS: Disabling LayerNorm affine parameters and the seven-logit classifier bias will retain at least 85% validation accuracy at 294,426,900 MACs while reducing learned parameters from 16,067 to 16,020.

INTENDED_EDIT: Make the input LayerNorm non-affine and the reference-class classifier bias-free, preserving the qualified 60-unit GRU, 25-frame schedule, and seven-logit formulation.

EVIDENCE: The non-affine LayerNorm design already qualified at 85.03% with 16,027 parameters and identical MACs. Removing only seven classifier biases is a smaller additional restriction than the failed recurrent-width, input-rank, or classifier-feature pruning experiments.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 7)
=======
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )
>>>>>>> REPLACE