MECHANISM: Minimal recurrent-width rebound

HYPOTHESIS: An 86-unit GRU trained at batch size 16 on frames 4–25 will achieve at least 85% validation accuracy while reducing total inference MACs to approximately 490.9 million.

INTENDED_EDIT: Reduce the GRU and classifier from 103 to 86 units, retain the qualified 22-frame schedule, and use batch size 16.

EVIDENCE: The 85-unit design missed qualification by only 0.092 percentage points at 480.6 million MACs, while the 91-unit design achieved 86.13%; testing the adjacent 86-unit width is the smallest capacity increase and cheapest unresolved frontier.

<<<<<<< SEARCH
BATCH_SIZE = 32
=======
BATCH_SIZE = 16
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(86, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE