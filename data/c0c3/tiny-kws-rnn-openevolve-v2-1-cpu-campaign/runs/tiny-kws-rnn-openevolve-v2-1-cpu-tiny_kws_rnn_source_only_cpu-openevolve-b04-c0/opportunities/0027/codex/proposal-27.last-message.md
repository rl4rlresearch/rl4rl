MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 67-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.6% versus the verified 68-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summary widths from 68 to 67 units and the classifier input from 204 to 201 features, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 68-unit model achieved 87.24% validation accuracy, 2.24 points above the requirement, and every one-unit reduction from 79 through 68 units qualified; this supports another minimal structural probe of the accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(204, 8)
=======
        self.gru = nn.GRU(20, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(201, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 68), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 67), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE