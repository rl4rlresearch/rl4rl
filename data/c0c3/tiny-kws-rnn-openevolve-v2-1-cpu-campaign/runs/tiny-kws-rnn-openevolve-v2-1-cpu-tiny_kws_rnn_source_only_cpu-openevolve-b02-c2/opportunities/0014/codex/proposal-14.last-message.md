MECHANISM: Dual-view recurrent pooling at the unresolved width boundary

HYPOTHESIS: A 96-unit GRU using concatenated sequence-mean and terminal states will achieve at least 85% validation accuracy with approximately 872,532,480 total inference MACs, 1.88% fewer than the qualified 97-unit model.

INTENDED_EDIT: Reduce recurrent width from 104 to 96 and expand the classifier input to combine mean-pooled and terminal recurrent representations.

EVIDENCE: The 97-unit dual-view model achieved 86.01% accuracy, and the 98-unit version achieved 87.12%; the prior 96-unit attempt timed out without producing accuracy evidence, leaving it as the closest unresolved cost boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        pooled = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        return self.classifier(pooled)
>>>>>>> REPLACE