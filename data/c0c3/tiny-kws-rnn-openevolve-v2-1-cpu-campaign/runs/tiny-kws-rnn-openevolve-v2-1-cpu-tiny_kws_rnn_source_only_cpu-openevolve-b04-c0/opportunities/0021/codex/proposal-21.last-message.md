MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 73-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.4% versus the verified 74-unit model.

INTENDED_EDIT: Reduce the GRU and all temporal summaries from 74 to 73 units and reduce the classifier input from 222 to 219 features, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 74-unit model achieved 86.13% validation accuracy, and every one-unit reduction from 79 through 74 units met the requirement, motivating another minimal structural probe of the accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 74, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)
=======
        self.gru = nn.GRU(20, 73, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(219, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 74, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 74, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 74), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 73, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 73, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 73), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE