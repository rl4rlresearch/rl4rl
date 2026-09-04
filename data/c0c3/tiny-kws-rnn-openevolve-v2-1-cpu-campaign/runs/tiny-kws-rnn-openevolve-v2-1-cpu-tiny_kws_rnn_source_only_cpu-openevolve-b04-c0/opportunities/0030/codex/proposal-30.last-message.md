MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 64-unit GRU with the established mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.7% versus the verified 65-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summary widths from 65 to 64 units and the classifier input from 195 to 192 features, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 65-unit model achieved 86.50% validation accuracy, 1.50 points above the requirement, and every tested one-unit reduction from 79 through 65 units qualified; this motivates the next minimal structural probe of the accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(195, 8)
=======
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 65, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 65), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 64), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE