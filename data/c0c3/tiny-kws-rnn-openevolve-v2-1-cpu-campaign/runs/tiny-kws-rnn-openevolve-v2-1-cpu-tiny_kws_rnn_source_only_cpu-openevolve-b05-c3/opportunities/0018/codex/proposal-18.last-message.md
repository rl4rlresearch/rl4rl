MECHANISM: One-unit width-boundary reduction of the dual-readout GRU

HYPOTHESIS: A 121-unit GRU with the qualified 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 848,734,480 to approximately 835,860,740.

INTENDED_EDIT: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 126 to 121 units while preserving the proven schedule and training procedure.

EVIDENCE: The adjacent 122-unit design qualified at 85.40% with 848,734,480 MACs, while the 123-unit design reached 86.01%; testing 121 units is the most informative remaining width-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(126, 8)
        self.endpoint_classifier = nn.Linear(126, 8)
=======
        self.gru = nn.GRU(20, 121, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(121, 8)
        self.endpoint_classifier = nn.Linear(121, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 121, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 121, device=device, dtype=dtype)
>>>>>>> REPLACE