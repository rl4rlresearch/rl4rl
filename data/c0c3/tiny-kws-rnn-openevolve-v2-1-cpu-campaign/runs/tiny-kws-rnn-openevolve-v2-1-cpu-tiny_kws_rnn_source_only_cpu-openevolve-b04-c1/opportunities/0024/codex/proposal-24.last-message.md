MECHANISM: One-unit recurrent-width boundary search

HYPOTHESIS: A 57-unit GRU with mean, maximum, and final readout will achieve at least 85% validation accuracy while reducing exact dense inference MACs by approximately 3% versus the verified 58-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 58 to 57 and resize its classifier and recurrent state tensors, preserving the successful 32-frame training and readout design.

EVIDENCE: The 58-unit model passed at 85.40% accuracy while the 56-unit model narrowly failed at 84.54%; testing the only remaining integer width directly identifies the structural capacity boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)
=======
        self.gru = nn.GRU(20, 57, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(171, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 58, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 57, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 57, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 57, device=device, dtype=dtype)
>>>>>>> REPLACE