MECHANISM: Endpoint-augmented one-unit width trim

HYPOTHESIS: A 94-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 428,624,800 to approximately 420,435,680.

INTENDED_EDIT: Use batch size 64, reduce the GRU and state width to 94, classify from the 188-feature mean/final-state concatenation, and process the established 16-frame even-index schedule.

EVIDENCE: The adjacent 95-unit endpoint-augmented design achieved 85.521% accuracy at 428,624,800 MACs; 94 units is the next untested structural reduction, while the failed 15-frame experiment supports preserving 16-frame temporal coverage.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 94, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(188, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 94, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 94, device=device, dtype=dtype)
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

<<<<<<< SEARCH
        return list(range(available_frames))
=======
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE