MECHANISM: Boundary-search recurrent width reduction

HYPOTHESIS: A 56-unit GRU with mean, maximum, and final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 5.9% versus the verified 58-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 58 to 56 and resize the classifier and recurrent state tensors, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 58-unit model achieved 85.40% accuracy at 355,092,240 MACs after the 60-unit model achieved 85.64%; both passed, and their modest accuracy change supports a final two-unit probe of the width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)
=======
        self.gru = nn.GRU(20, 56, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 58, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 56, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 56, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 56, device=device, dtype=dtype)
>>>>>>> REPLACE