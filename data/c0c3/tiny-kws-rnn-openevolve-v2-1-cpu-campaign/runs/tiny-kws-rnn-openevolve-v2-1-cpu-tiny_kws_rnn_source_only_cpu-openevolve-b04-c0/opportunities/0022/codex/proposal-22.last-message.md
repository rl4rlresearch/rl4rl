MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 72-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact recurrent MACs by approximately 2.4% versus the verified 73-unit model.

INTENDED_EDIT: Reduce the GRU and all temporal summaries from 73 to 72 units and reduce the classifier input from 219 to 216 features, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 73-unit model achieved 87.36% validation accuracy, 2.36 points above the requirement, and every one-unit reduction from 79 through 73 units qualified; this supports another minimal structural probe of the accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 73, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(219, 8)
=======
        self.gru = nn.GRU(20, 72, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(216, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 73, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 73, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 73), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 72, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 72, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 72), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE