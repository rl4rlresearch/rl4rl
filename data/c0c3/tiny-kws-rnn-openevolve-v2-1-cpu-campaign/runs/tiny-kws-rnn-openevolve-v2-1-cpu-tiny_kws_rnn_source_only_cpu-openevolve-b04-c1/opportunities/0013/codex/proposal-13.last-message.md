MECHANISM: Boundary-search recurrent width reduction

HYPOTHESIS: An 86-unit GRU with mean-plus-final readout will achieve at least 85% validation accuracy while reducing dense inference MACs by approximately 4% versus the verified 88-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 88 to 86 and resize the classifier and recurrent state tensors, preserving all 32 frames and the established training procedure.

EVIDENCE: The 88-unit model achieved 85.77% accuracy, while 84 units narrowly missed at 84.79%; testing the midpoint directly resolves the remaining width boundary with a meaningful structural cost reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(172, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE