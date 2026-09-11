MECHANISM: Adjacent-width clocked dual-timescale GRU

HYPOTHESIS: A 76-unit fast GRU over all 20 scheduled frames plus a 76-unit slow GRU over four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 482,711,460 to approximately 471,239,520.

INTENDED_EDIT: Reduce both recurrent widths from 77 to 76 units and resize the three-summary classifier input from 231 to 228 features, preserving the qualified schedule and training procedure.

EVIDENCE: The adjacent 77-unit design achieved 86.75% validation accuracy at 482,711,460 MACs, leaving a 1.75-point margin after every tested 77–80-unit dual-timescale design qualified; 76 units is the most informative remaining capacity-boundary probe.

<<<<<<< SEARCH
        self.fast_cell = nn.GRUCell(20, 77)
        self.slow_norm = nn.LayerNorm(77)
        self.slow_cell = nn.GRUCell(77, 77)
        self.classifier = nn.Linear(231, 8)
=======
        self.fast_cell = nn.GRUCell(20, 76)
        self.slow_norm = nn.LayerNorm(76)
        self.slow_cell = nn.GRUCell(76, 76)
        self.classifier = nn.Linear(228, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fast = torch.zeros(batch_size, 77, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 77, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 77, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 77, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 77, device=device, dtype=dtype)
=======
        fast = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
>>>>>>> REPLACE