MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 79-unit GRU with the proven mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by about 2.2% versus the verified 80-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summary widths from 80 to 79 units and the classifier input from 240 to 237 features, preserving all 32 frames and the established training procedure.

EVIDENCE: The 80-unit model achieved 85.64% accuracy, while the 82-unit model achieved 86.75%; a one-unit reduction is the smallest structural probe of the accuracy-cost boundary after larger two-unit reductions showed measurable accuracy sensitivity.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
=======
        self.gru = nn.GRU(20, 79, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(237, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 80), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 79, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 79, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 79), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE