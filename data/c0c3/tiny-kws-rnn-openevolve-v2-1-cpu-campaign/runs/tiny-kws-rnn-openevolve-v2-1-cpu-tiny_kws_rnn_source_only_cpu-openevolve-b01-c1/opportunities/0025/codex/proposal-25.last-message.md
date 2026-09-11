MECHANISM: High-frequency boundary-band pruning

HYPOTHESIS: Dropping only the highest log-mel band will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 0.9% versus the verified 20-band, 28-step model.

INTENDED_EDIT: Reduce the GRU input width from 20 to 19 and feed bands 0–18 through both recurrent execution paths, preserving hidden width, temporal schedule, classifier, and training.

EVIDENCE: Dropping the lowest band narrowly missed the target at 84.79%, showing that single-band structural pruning is near the feasible boundary; testing the opposite spectral boundary is the most direct way to determine whether the lowest-frequency information was specifically load-bearing.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
=======
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 92, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(frame[:, :19]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[:, :, :19]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE