MECHANISM: One-unit width reduction of the qualified dual-readout GRU

HYPOTHESIS: A 126-unit GRU with the proven 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 914,570,180 to approximately 901,207,440.

INTENDED_EDIT: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 127 to 126 units while preserving the qualified 20-frame schedule and training procedure.

EVIDENCE: The current 127-unit design achieved 86.26% accuracy, providing a 1.26-point qualification margin; the dual-readout mechanism also qualified at 128 units, so testing the adjacent 126-unit width is the most direct next cost-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(127, 8)
        self.endpoint_classifier = nn.Linear(127, 8)
=======
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(126, 8)
        self.endpoint_classifier = nn.Linear(126, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
>>>>>>> REPLACE