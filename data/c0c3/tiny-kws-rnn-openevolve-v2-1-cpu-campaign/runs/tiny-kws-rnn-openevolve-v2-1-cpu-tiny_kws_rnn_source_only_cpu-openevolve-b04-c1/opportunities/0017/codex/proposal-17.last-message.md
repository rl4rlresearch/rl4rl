MECHANISM: Continued width reduction with salience-preserving temporal pooling

HYPOTHESIS: A 72-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 9.2% versus the verified 76-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 76 to 72 and resize the classifier and recurrent state tensors, preserving all 32 causal frames and the established three-way readout.

EVIDENCE: The 84-, 80-, and 76-unit max-augmented models all exceeded 85% accuracy; the 76-unit result retained 1.38 percentage points of headroom while achieving the lowest verified inference cost, supporting the next four-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 76, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(228, 8)
=======
        self.gru = nn.GRU(20, 72, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(216, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 76, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 72, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 72, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 72, device=device, dtype=dtype)
>>>>>>> REPLACE