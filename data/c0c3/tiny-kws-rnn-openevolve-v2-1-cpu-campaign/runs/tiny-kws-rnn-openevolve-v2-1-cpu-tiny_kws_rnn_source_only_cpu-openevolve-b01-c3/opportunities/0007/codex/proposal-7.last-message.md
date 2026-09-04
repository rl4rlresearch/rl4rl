MECHANISM: Further width trimming of the proven 16-step causal GRU

HYPOTHESIS: A 116-unit GRU trained with batch size 64 on the proven even-index schedule will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6% versus the qualified 120-unit design.

INTENDED_EDIT: Reduce recurrent, summary-state, and classifier width from 120 to 116 units while preserving the successful training regime and 16-frame schedule.

EVIDENCE: The 120-unit model achieved 86.01% accuracy—1.01 points above the requirement—and improved upon the qualified 124-unit model despite lower width, providing margin for another four-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
=======
        self.gru = nn.GRU(20, 116, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 116, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 116, device=device, dtype=dtype)
>>>>>>> REPLACE