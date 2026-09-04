MECHANISM: Adjacent width-boundary reduction of the dual-readout GRU

HYPOTHESIS: A 117-unit GRU with the qualified 20-frame mean/endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 797,826,320 to approximately 785,343,780.

INTENDED_EDIT: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 122 to 117 units while preserving the qualified frame schedule and training procedure.

EVIDENCE: The adjacent 118-unit design qualified at 85.03% with 797,826,320 MACs, and every tested 118–126-unit dual-readout model qualified; testing 117 units is the most direct remaining cost-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 122, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(122, 8)
        self.endpoint_classifier = nn.Linear(122, 8)
=======
        self.gru = nn.GRU(20, 117, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(117, 8)
        self.endpoint_classifier = nn.Linear(117, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 122, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 122, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 117, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 117, device=device, dtype=dtype)
>>>>>>> REPLACE