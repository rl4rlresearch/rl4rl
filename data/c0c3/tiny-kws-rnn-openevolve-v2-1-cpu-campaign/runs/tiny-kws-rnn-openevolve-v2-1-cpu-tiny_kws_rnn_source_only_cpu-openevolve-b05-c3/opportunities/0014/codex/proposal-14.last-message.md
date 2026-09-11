MECHANISM: One-unit boundary width reduction of the qualified dual-readout GRU

HYPOTHESIS: A 125-unit GRU with the proven 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 126-unit design’s 901,207,440 MACs.

INTENDED_EDIT: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 128 to 125 units while preserving the qualified 20-frame schedule and training procedure.

EVIDENCE: The 126-unit 20-frame dual-readout model achieved 86.87% accuracy, a 1.87-point margin, while successively tested 128- and 127-unit variants also qualified; the adjacent 125-unit width is the most direct next cost-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
        self.endpoint_classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(125, 8)
        self.endpoint_classifier = nn.Linear(125, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 125, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 125, device=device, dtype=dtype)
>>>>>>> REPLACE