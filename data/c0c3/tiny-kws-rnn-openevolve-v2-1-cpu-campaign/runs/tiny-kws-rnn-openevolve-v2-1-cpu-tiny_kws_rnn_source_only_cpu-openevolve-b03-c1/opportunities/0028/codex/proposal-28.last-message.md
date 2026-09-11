MECHANISM: Temporal-max-assisted width interpolation

HYPOTHESIS: A 78-unit GRU retaining the successful 27-step endpoint, mean, and temporal-maximum readout will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 4.4%.

INTENDED_EDIT: Reduce the GRU hidden width from 80 to 78 and resize its recurrent states and classifier input from 240 to 234 features.

EVIDENCE: The 80-unit temporal-maximum model achieved 87.61%, while the 76-unit version narrowly missed at 84.79%; testing their midpoint locates the capacity boundary for this specific readout at lower cost.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
=======
        self.gru = nn.GRU(20, 78, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(234, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 80, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 78, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 78, device=device, dtype=dtype)
>>>>>>> REPLACE