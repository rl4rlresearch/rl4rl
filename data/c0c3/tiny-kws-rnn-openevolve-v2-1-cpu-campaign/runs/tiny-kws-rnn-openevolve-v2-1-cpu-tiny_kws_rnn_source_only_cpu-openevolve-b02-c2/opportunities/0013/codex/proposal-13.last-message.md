MECHANISM: Dual-view recurrent pooling with one-unit width refinement

HYPOTHESIS: A 97-unit GRU classifying concatenated mean and terminal outputs will achieve at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.9% versus the qualified 98-unit dual-view model.

INTENDED_EDIT: Reduce recurrent width from 99 to 97 and replace mean-only classification with a linear classifier over concatenated sequence-mean and terminal recurrent states.

EVIDENCE: The 98-unit dual-view model achieved 87.12% accuracy and 906,045,280 total MACs, substantially outperforming the same-width mean-only model’s 84.54%; the 96-unit dual-view attempt produced no accuracy evidence because training timed out, making 97 units the closest informative capacity reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
=======
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
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