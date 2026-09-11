MECHANISM: Boundary-seeking recurrent width reduction

HYPOTHESIS: A 106-unit GRU with the proven batch-64 training regime and 16-frame even-index schedule will retain at least 85% validation accuracy while reducing total inference MACs by approximately 3.4% versus the qualified 108-unit design.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier width from 116 to 106 units while preserving the validated training procedure and frame schedule.

EVIDENCE: The 108-unit design qualified at 85.399% accuracy and 541,499,040 total MACs after the 110-unit design qualified at 86.012%; this supports another two-unit boundary test, projected near 523,177,840 total MACs.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 116, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
=======
        self.gru = nn.GRU(20, 106, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(106, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 116, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 116, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 106, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 106, device=device, dtype=dtype)
>>>>>>> REPLACE