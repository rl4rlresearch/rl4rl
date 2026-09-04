MECHANISM: Variance-assisted recurrent-width pruning

HYPOTHESIS: A 64-unit, 27-step GRU retaining the temporal-deviation readout will achieve at least 85% validation accuracy while reducing dense inference MACs versus the verified 65-unit model.

INTENDED_EDIT: Reduce the GRU and all recurrent summary widths from 65 to 64 units, and resize the four-statistic classifier input from 260 to 256 features.

EVIDENCE: Temporal deviation raised the 65-unit, 27-step design from 84.79% to 86.50%; testing it at 64 units directly probes whether that accuracy gain permits the next structural width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(260, 8)
=======
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 65, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 65), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 64), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE