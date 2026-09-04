MECHANISM: Elementwise mean–terminal interaction readout

HYPOTHESIS: A 124-unit GRU augmented with mean-times-terminal features will recover at least two validation examples, reaching 85% accuracy while keeping total inference below the qualified 125-unit model’s 1.198B MACs.

INTENDED_EDIT: Reduce recurrent width from 126 to 124 and extend the classifier with a parameter-free multiplicative interaction between mean and terminal recurrent states.

EVIDENCE: The otherwise identical 124-unit dual-readout model missed qualification by only two examples at 84.79% with strong 0.3884 cross-entropy; adding nonlinear temporal interaction costs only 992 additional classifier MACs per example while preserving most of its recurrent savings.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
=======
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(372, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, count = state
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=-1
        )
        return self.classifier(features)
=======
        hidden, summary, count = state
        mean = summary / count.clamp_min(1.0)
        terminal = hidden[:, 0, :]
        features = torch.cat((mean, terminal, mean * terminal), dim=-1)
        return self.classifier(features)
>>>>>>> REPLACE