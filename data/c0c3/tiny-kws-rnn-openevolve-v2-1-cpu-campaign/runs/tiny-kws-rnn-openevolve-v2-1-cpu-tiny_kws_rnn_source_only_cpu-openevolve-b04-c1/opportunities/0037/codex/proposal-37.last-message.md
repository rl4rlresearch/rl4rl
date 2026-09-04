MECHANISM: Lossless zero-mean mel coordinate elimination

HYPOTHESIS: Representing each non-affine layer-normalized frame with its first 19 coordinates will retain at least 85% accuracy while reducing total inference MACs by approximately 3.55M, because the omitted coordinate is determined by the normalization’s zero-sum constraint.

INTENDED_EDIT: Disable redundant LayerNorm affine parameters, reduce the GRU input width from 20 to 19, and slice one normalized coordinate in both recurrent execution paths.

EVIDENCE: Directly fusing two high-frequency bands at the same 19-input GRU width reached only 84.54%, indicating that its information loss was harmful; this patch achieves the same recurrent MAC reduction without discarding spectral information.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)
=======
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(19, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)
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