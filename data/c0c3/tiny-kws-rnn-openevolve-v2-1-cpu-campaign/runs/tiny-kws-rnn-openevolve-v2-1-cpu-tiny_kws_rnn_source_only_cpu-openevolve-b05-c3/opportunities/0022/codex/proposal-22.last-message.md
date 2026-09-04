MECHANISM: Adjacent width-boundary reduction of the dual-readout GRU

HYPOTHESIS: A 118-unit GRU with the qualified 20-frame mean/endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 810,406,660 to approximately 797,826,320.

INTENDED_EDIT: Reduce the recurrent hidden state and both classifier input widths from 119 to 118 units while preserving the proven frame schedule and training procedure.

EVIDENCE: The adjacent 119-unit design achieved 85.89% accuracy, and every tested dual-readout width from 119 through 126 qualified; 118 units is therefore the most informative next cost-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 119, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)
        self.endpoint_classifier = nn.Linear(119, 8)
=======
        self.gru = nn.GRU(20, 118, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(118, 8)
        self.endpoint_classifier = nn.Linear(118, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 119, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 118, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 118, device=device, dtype=dtype)
>>>>>>> REPLACE