MECHANISM: Recurrent-width scaling with dual temporal readout

HYPOTHESIS: A 96-unit GRU retaining all 32 frames and the mean-plus-final readout will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 13.6% versus the verified 104-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 104 to 96 and resize the classifier and recurrent summary state accordingly.

EVIDENCE: The 104-unit full-sequence model achieved 88.34% accuracy, providing a 3.34-point margin above the requirement; prior temporal subsampling was less reliable, so reducing width while preserving all temporal evidence is the best-supported next cost reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(208, 8)
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