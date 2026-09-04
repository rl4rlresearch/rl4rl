MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 78-unit GRU with the proven mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by roughly 2.3% versus the verified 79-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summaries from 79 to 78 units and the classifier input from 237 to 234 features, preserving all 32 frames and the established training procedure.

EVIDENCE: The 79-unit model achieved 87.24% accuracy, 2.24 points above the requirement, and outperformed the qualifying 80-unit model; this supports another minimal one-unit structural reduction at the current accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 79, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(237, 8)
=======
        self.gru = nn.GRU(20, 78, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(234, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 79, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 79, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 79), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 78, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 78), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE