MECHANISM: Temporal-max-assisted recurrent width reduction

HYPOTHESIS: An 82-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs relative to the verified 84-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summaries from 84 to 82 units and the classifier input from 252 to 246 features, preserving all 32 frames and the proven training procedure.

EVIDENCE: The 84-unit triple-readout model achieved 87.12% accuracy, 2.12 points above the requirement, after the same readout enabled both 86- and 84-unit models to qualify; this supports another conservative two-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
=======
        self.gru = nn.GRU(20, 82, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(246, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 84), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 82, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 82, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 82), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE