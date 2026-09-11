MECHANISM: One-unit width reduction of the endpoint-aware dual-readout GRU

HYPOTHESIS: A 123-unit GRU with the qualified 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 874,775,360 to approximately 861,706,020.

INTENDED_EDIT: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 124 to 123 units while preserving the schedule and training procedure.

EVIDENCE: The adjacent 124- and 125-unit designs both achieved 85.40% accuracy, while the 126-unit design achieved 86.87%; testing 123 units is the most direct remaining cost-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(124, 8)
        self.endpoint_classifier = nn.Linear(124, 8)
=======
        self.gru = nn.GRU(20, 123, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(123, 8)
        self.endpoint_classifier = nn.Linear(123, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 123, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 123, device=device, dtype=dtype)
>>>>>>> REPLACE