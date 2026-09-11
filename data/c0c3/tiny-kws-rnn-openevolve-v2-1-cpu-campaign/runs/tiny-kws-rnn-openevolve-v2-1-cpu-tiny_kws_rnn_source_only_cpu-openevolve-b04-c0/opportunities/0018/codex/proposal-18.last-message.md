MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 76-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.3% versus the verified 77-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summaries from 77 to 76 units and the classifier input from 231 to 228 features, preserving all 32 frames and the established training procedure.

EVIDENCE: The 77-unit model achieved 86.87% accuracy, 1.87 points above the requirement, after the 78- and 79-unit models also qualified; this supports another minimal structural reduction at the current accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 77, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(231, 8)
=======
        self.gru = nn.GRU(20, 76, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(228, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 77, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 77, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 77), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 76), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE