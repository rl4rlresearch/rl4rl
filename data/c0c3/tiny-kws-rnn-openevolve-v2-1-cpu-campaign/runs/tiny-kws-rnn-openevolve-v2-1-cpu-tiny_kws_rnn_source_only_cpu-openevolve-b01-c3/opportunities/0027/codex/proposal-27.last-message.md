MECHANISM: Endpoint-augmented one-unit width trim

HYPOTHESIS: A 95-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 436,892,160 to approximately 428,624,800.

INTENDED_EDIT: Reduce the recurrent state and temporal summary from 101 to 95 units and replace the mean-only classifier with a single linear classifier over the 190-feature concatenation of temporal mean and final hidden state.

EVIDENCE: The qualified 96-unit endpoint-augmented design achieved 85.276% accuracy at 436,892,160 MACs; 95 units is the next untested structural reduction and preserves the 16-frame coverage shown to be important.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 101, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(101, 8)
=======
        self.gru = nn.GRU(20, 95, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(190, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 101, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 101, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 95, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 95, device=device, dtype=dtype)
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