MECHANISM: One-unit width reduction of the qualified dual-readout GRU

HYPOTHESIS: A 122-unit GRU with the proven 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 861,706,020 to approximately 848,734,480.

INTENDED_EDIT: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 125 to 122 units while preserving the qualified schedule and training procedure.

EVIDENCE: The adjacent 123-unit design achieved 86.01% accuracy at 861,706,020 MACs, leaving a 1.01-point margin and making 122 units the most informative next width-boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(125, 8)
        self.endpoint_classifier = nn.Linear(125, 8)
=======
        self.gru = nn.GRU(20, 122, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(122, 8)
        self.endpoint_classifier = nn.Linear(122, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 125, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 125, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 122, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 122, device=device, dtype=dtype)
>>>>>>> REPLACE