MECHANISM: Upper-edge spectral input pruning

HYPOTHESIS: Omitting the highest-frequency mel band will retain at least 85% validation accuracy on the qualified 25-step design while reducing total inference MACs from 294,426,900 to approximately 290,759,400.

INTENDED_EDIT: Adopt the qualified non-affine normalization, bias-free seven-logit head, and endpoint-preserving schedule, while reducing the GRU input to the lowest 19 mel bands.

EVIDENCE: The full 20-band design achieved 85.52% accuracy, while removing the lowest-frequency band narrowly missed at 84.42%. Testing the opposite spectral edge determines whether that miss was band-specific rather than caused by 19-band capacity.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
=======
        self.input_norm = nn.LayerNorm(19, elementwise_affine=False)
        self.gru = nn.GRU(
            19, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(frame[..., :-1]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., :-1]),
            hidden.transpose(0, 1).contiguous(),
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(features)
=======
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(3, available_frames - 1))
=======
        return list(range(3, available_frames - 5)) + [available_frames - 4]
>>>>>>> REPLACE