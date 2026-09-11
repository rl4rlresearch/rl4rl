MECHANISM: Boundary-search recurrent width reduction

HYPOTHESIS: A 58-unit GRU using mean, maximum, and final recurrent outputs will achieve at least 85% validation accuracy while reducing exact dense inference MACs by approximately 5.7% versus the verified 60-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 60 to 58 and resize the classifier and recurrent state tensors, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 60-unit model achieved 85.64% accuracy at 376,725,600 MACs, while every tested max-augmented width from 60 through 84 passed; a two-unit reduction cautiously probes the remaining accuracy boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(180, 8)
=======
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 60, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 60, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 58, device=device, dtype=dtype)
>>>>>>> REPLACE