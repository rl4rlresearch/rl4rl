MECHANISM: Endpoint-augmented one-unit width trim

HYPOTHESIS: A 96-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 445,237,760 to approximately 436,892,160.

INTENDED_EDIT: Reduce the recurrent state and temporal summary to 96 units and classify from the concatenated temporal mean and final hidden state.

EVIDENCE: The qualified 97-unit endpoint-augmented design achieved 86.135% accuracy at 445,237,760 MACs, providing a 1.135-point margin for testing the next one-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)
>>>>>>> REPLACE