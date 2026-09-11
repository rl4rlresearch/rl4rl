MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 69-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2.5% versus the verified 70-unit model.

INTENDED_EDIT: Reduce the GRU and all recurrent summary widths from 70 to 69 units and reduce the classifier input from 210 to 207 features, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 70-unit model achieved 87.12% validation accuracy, 2.12 points above the requirement, and every one-unit reduction from 79 through 70 units qualified; this supports another minimal structural probe at the current accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 70, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(210, 8)
=======
        self.gru = nn.GRU(20, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(207, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 70, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 70, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 70), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 69), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE