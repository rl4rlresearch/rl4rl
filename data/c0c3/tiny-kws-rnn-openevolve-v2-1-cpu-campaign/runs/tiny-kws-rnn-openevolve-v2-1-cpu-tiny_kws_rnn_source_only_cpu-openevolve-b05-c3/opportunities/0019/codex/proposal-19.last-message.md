MECHANISM: One-unit width-boundary reduction of the qualified dual-readout GRU

HYPOTHESIS: A 120-unit GRU with the qualified 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 835,860,740 to approximately 823,084,800.

INTENDED_EDIT: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 121 to 120 units while preserving the proven schedule and training procedure.

EVIDENCE: The adjacent 121-unit design achieved 85.40% accuracy at 835,860,740 MACs, and every tested 121–126-unit dual-readout design qualified; testing 120 units is the most informative remaining width-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 121, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(121, 8)
        self.endpoint_classifier = nn.Linear(121, 8)
=======
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
        self.endpoint_classifier = nn.Linear(120, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 121, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 121, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
>>>>>>> REPLACE