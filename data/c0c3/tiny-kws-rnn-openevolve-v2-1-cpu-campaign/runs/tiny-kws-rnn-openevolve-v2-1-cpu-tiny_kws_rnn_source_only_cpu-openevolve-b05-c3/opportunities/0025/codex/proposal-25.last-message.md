MECHANISM: Adjacent width-boundary reduction of the dual-readout GRU

HYPOTHESIS: A 116-unit GRU with the qualified 20-frame mean/endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 785,343,780 to approximately 772,959,040.

INTENDED_EDIT: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 117 to 116 units while preserving the qualified schedule and training procedure.

EVIDENCE: The adjacent 117-unit design achieved 85.52% accuracy at 785,343,780 MACs, and every tested 117–126-unit dual-readout design qualified; testing 116 units is the most direct remaining cost-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 117, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(117, 8)
        self.endpoint_classifier = nn.Linear(117, 8)
=======
        self.gru = nn.GRU(20, 116, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
        self.endpoint_classifier = nn.Linear(116, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 117, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 117, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 116, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 116, device=device, dtype=dtype)
>>>>>>> REPLACE