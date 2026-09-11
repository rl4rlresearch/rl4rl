MECHANISM: Recurrent width reduction with dual-timescale readout

HYPOTHESIS: A 92-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 7.5% versus the verified 96-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 96 to 92 and resize the classifier and recurrent state tensors accordingly, while preserving all 32 frames and the successful dual-timescale readout.

EVIDENCE: The 96-unit dual-readout model achieved 85.89% accuracy with 872,532,480 total MACs, improving on the 100-unit model’s accuracy despite its smaller width; this provides 0.89 percentage points of headroom for another four-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
=======
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
>>>>>>> REPLACE