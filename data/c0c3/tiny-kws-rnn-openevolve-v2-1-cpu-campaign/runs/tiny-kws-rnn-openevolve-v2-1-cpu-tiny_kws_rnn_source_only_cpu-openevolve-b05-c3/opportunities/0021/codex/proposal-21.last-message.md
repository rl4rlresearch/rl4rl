MECHANISM: Adjacent width-boundary reduction of the dual-readout GRU

HYPOTHESIS: A 119-unit GRU with the qualified 20-frame mean/endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 823,084,800 to approximately 810,406,660.

INTENDED_EDIT: Reduce the recurrent hidden state and both classifier input widths from 123 to 119 units while preserving the proven schedule and training procedure.

EVIDENCE: The 120-unit dual-readout model achieved 86.26% accuracy at 823,084,800 MACs, and every tested width from 120 through 126 qualified; testing the adjacent 119-unit width is the most informative remaining cost-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 123, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(123, 8)
        self.endpoint_classifier = nn.Linear(123, 8)
=======
        self.gru = nn.GRU(20, 119, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)
        self.endpoint_classifier = nn.Linear(119, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 123, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 123, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 119, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
>>>>>>> REPLACE