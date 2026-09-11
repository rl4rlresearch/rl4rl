MECHANISM: Boundary-seeking width trim of the qualified 16-step causal GRU

HYPOTHESIS: A 110-unit GRU trained with batch size 64 on the proven even-index schedule will retain at least 85% validation accuracy while reducing total inference MACs by approximately 3.3% versus the qualified 112-unit design.

INTENDED_EDIT: Reproduce the successful batch-64 training regime and reduce recurrent, summary-state, and classifier width from 128 to 110 units while retaining all 16 scheduled frames.

EVIDENCE: The 112-unit design achieved 85.644% accuracy at 579,080,320 total MACs; a two-unit reduction is a conservative boundary test expected to cost about 560,133,200 MACs while using its remaining 0.644-point accuracy margin.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(110, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
>>>>>>> REPLACE