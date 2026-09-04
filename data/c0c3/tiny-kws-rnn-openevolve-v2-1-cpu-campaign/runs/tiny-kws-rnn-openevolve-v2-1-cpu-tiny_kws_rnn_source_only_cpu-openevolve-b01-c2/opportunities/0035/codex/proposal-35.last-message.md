MECHANISM: Iterative late-frame pruning with endpoint preservation

HYPOTHESIS: Processing frames 3–25 plus frame 28 will retain at least 85% validation accuracy while reducing total inference MACs from 294,426,900 to approximately 282,690,900.

INTENDED_EDIT: Adopt the qualified non-affine normalization and bias-free seven-logit head, then remove frame 26 from the proven endpoint-preserving schedule for 24 recurrent steps.

EVIDENCE: The bias-free 25-step reference achieved 85.52% accuracy, and removing frame 27 while preserving frame 28 previously qualified; pruning the next-latest contiguous frame is the most direct test of further temporal redundancy without disrupting the important endpoint or early continuity.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
=======
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
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

<<<<<<< SEARCH
        return list(range(3, available_frames - 5)) + [available_frames - 4]
=======
        return list(range(3, available_frames - 6)) + [available_frames - 4]
>>>>>>> REPLACE