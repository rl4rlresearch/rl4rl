MECHANISM: LayerNorm zero-sum input-coordinate elimination

HYPOTHESIS: Removing one algebraically redundant affine-free LayerNorm coordinate will retain at least 85% validation accuracy while reducing every GRU step by 300 MACs.

INTENDED_EDIT: Disable the absorbable LayerNorm affine parameters, pass 19 normalized bands to the 100-unit GRU, and preserve all 32 causal steps.

EVIDENCE: The full-rate 100-unit model achieved 85.03%, while temporal and output-head reductions failed; affine-free LayerNorm outputs sum to zero, so one coordinate is recoverable from the other 19 and its GRU contribution can be represented by the remaining input weights and biases.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
=======
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(19, 100, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(frame)[..., :19].unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames)[..., :19], hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE