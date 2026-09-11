MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 74-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.4% versus the verified 75-unit model.

INTENDED_EDIT: Reduce the GRU and all temporal summaries from 75 to 74 units and reduce the classifier input from 225 to 222 features, while preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 75-unit model achieved 86.50% validation accuracy, and every one-unit reduction from 79 through 75 units qualified; this supports one more minimal structural probe of the accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 75, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(225, 8)
=======
        self.gru = nn.GRU(20, 74, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 75, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 75, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 75), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 74, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 74, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 74), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE