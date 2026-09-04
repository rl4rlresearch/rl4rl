MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 66-unit GRU with the established mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.6% versus the verified 67-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summary widths from 67 to 66 units and the classifier input from 201 to 198 features, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 67-unit model achieved 86.87% validation accuracy, 1.87 points above the requirement, and every one-unit reduction from 79 through 67 units qualified; this supports another minimal structural probe of the current accuracy-cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(201, 8)
=======
        self.gru = nn.GRU(20, 66, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(198, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 67), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 66, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 66), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE