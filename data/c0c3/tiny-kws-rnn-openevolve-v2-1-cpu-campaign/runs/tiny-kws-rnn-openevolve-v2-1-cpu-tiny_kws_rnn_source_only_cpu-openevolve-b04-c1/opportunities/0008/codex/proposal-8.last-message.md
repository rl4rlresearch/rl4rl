MECHANISM: Recurrent width reduction with dual-timescale readout

HYPOTHESIS: A 96-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing dense inference MACs by approximately 7% versus the verified 100-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 100 to 96 and resize the classifier and recurrent state tensors accordingly, preserving all 32 frames and the successful dual-timescale readout.

EVIDENCE: The 100-unit dual-readout model achieved 85.64% accuracy with 940,184,000 MACs, outperforming the wider 104-unit model’s 85.03%; this suggests the enriched readout provides enough accuracy headroom to test another modest structural width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(200, 8)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE