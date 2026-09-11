MECHANISM: Hardware-friendly temporal-max-assisted width reduction

HYPOTHESIS: An 80-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by about 8.4% versus the verified 84-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summaries from 84 to 80 units and the classifier input from 252 to 240 features, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 82-unit model reached 86.75% accuracy despite exceeding the training-time limit, while the verified 84-unit model reached 87.12%; this indicates the triple readout retains sufficient accuracy at smaller widths and motivates testing the aligned 80-unit width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
=======
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 84), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 80), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE