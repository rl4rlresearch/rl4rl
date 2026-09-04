MECHANISM: Single edge-band recurrent-input pruning

HYPOTHESIS: Removing the lowest-frequency mel band from the proven 60-unit, 25-step, seven-logit design will retain at least 85% validation accuracy while reducing total inference MACs from 294,426,900 to approximately 290,759,400.

INTENDED_EDIT: Adopt the qualified non-affine normalization and bias-free seven-logit head, then reduce the GRU input from 20 to 19 features by omitting the lowest-frequency normalized mel band.

EVIDENCE: The 20-band reference design achieved 85.52% accuracy at 294,426,900 MACs, while the more aggressive rank-17 bottleneck narrowly missed at 84.42%; pruning only one direct spectral input is the smallest structural probe between them.

<<<<<<< SEARCH
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
=======
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(
            19, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        output, hidden = self.gru(
            self.input_norm(frame)[..., 1:].unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        outputs, hidden = self.gru(
            self.input_norm(frames)[..., 1:],
            hidden.transpose(0, 1).contiguous(),
        )
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