MECHANISM: Recurrent width reduction with dual-timescale readout

HYPOTHESIS: An 88-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 7.8% versus the verified 92-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 92 to 88 and resize the classifier and recurrent state tensors accordingly, preserving all 32 frames and the successful dual-timescale readout.

EVIDENCE: The 92-unit model achieved 87.12% accuracy with 807,384,640 MACs—its best observed accuracy among the width-reduction sequence and 2.12 percentage points above threshold—supporting another modest four-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE