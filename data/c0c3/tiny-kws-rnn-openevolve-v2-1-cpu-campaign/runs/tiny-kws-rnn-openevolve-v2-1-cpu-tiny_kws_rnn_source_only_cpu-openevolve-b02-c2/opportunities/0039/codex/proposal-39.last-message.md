MECHANISM: Single-layer reference-class logit parameterization

HYPOTHESIS: The qualified 84-unit, 18-band GRU with seven learned class-contrast logits and one fixed reference logit will retain at least 85% accuracy while reducing total inference MACs from 608,609,400 to approximately 608,472,480.

INTENDED_EDIT: Adopt the qualified 18-band recurrent trunk and replace the 168→8 classifier with one 168→7 linear layer whose outputs are concatenated with a zero reference-class logit.

EVIDENCE: The 84-unit, 18-band trunk achieved 85.03% accuracy. The failed 168→7→8 head used a non-convex two-matrix factorization; directly learning seven logits preserves the full eight-class softmax function class without that factorization and saves 168 MACs per example.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
        self.input_norm = nn.LayerNorm(18)
        self.gru = nn.GRU(18, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        output, hidden = self.gru(
            self.input_norm(frame[..., :18]).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        outputs, hidden = self.gru(
            self.input_norm(frames[..., :18]), hidden.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        return self.classifier(pooled)
=======
        pooled = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        contrasts = self.classifier(pooled)
        reference = contrasts.new_zeros((contrasts.shape[0], 1))
        return torch.cat((contrasts, reference), dim=1)
>>>>>>> REPLACE