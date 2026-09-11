MECHANISM: Dual-timescale readout with narrower recurrent state

HYPOTHESIS: A 90-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 906,045,280 to approximately 775,749,600.

INTENDED_EDIT: Reduce the qualified dual-readout GRU width from 98 to 90 while preserving all 32 causal frames and expand the classifier input to the concatenated 180-dimensional mean/final representation.

EVIDENCE: The 98-unit dual-timescale model achieved 87.12% accuracy versus 84.54% for the 98-unit mean-only model, showing a 2.58-point gain and enough margin to test a meaningful eight-unit structural reduction without revisiting failed temporal subsampling.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
=======
        self.gru = nn.GRU(20, 90, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(180, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 90, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 90, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))
>>>>>>> REPLACE