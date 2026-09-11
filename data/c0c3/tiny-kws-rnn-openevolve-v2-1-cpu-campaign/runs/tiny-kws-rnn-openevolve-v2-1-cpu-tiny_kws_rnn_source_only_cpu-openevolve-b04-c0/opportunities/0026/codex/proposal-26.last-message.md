MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 68-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.6% versus the verified 69-unit model.

INTENDED_EDIT: Reduce the GRU and all recurrent summary widths from 69 to 68 units and reduce the classifier input from 207 to 204 features, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 69-unit model achieved 87.48% validation accuracy, 2.48 points above the requirement, and every one-unit reduction from 79 through 69 units qualified; this supports another minimal structural probe at the current accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(207, 8)
=======
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(204, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 69), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 68), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE