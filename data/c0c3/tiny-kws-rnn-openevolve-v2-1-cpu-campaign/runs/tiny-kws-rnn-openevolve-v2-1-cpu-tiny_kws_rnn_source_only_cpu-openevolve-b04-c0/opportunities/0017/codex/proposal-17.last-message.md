MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 77-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.3% versus the verified 78-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summary widths from 78 to 77 units and the classifier input from 234 to 231 features, preserving all 32 frames and the established training procedure.

EVIDENCE: The 78-unit model achieved 86.13% accuracy, 1.13 points above the requirement, after both 79- and 78-unit one-unit reductions qualified; this motivates one more minimal structural probe of the accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 78, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(234, 8)
=======
        self.gru = nn.GRU(20, 77, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(231, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 78, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 78), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 77, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 77, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 77), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE