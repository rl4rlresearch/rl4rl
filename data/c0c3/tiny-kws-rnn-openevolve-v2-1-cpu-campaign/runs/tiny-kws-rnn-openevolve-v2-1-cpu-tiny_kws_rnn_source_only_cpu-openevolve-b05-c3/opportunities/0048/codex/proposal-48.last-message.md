MECHANISM: Adjacent-width clocked dual-timescale GRU

HYPOTHESIS: A 73-unit fast GRU over 20 frames plus a 73-unit slow GRU over four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 448,706,400 to approximately 437,645,220.

INTENDED_EDIT: Reduce both recurrent widths from 80 to 73 units and resize the three-summary classifier input from 240 to 219 features, preserving the qualified schedule and training procedure.

EVIDENCE: The adjacent 74-unit design achieved 86.87% validation accuracy at 448,706,400 MACs, and every tested 74–80-unit dual-timescale design qualified; 73 units is the most informative remaining capacity-boundary probe.

<<<<<<< SEARCH
        self.fast_cell = nn.GRUCell(20, 80)
        self.slow_norm = nn.LayerNorm(80)
        self.slow_cell = nn.GRUCell(80, 80)
        self.classifier = nn.Linear(240, 8)
=======
        self.fast_cell = nn.GRUCell(20, 73)
        self.slow_norm = nn.LayerNorm(73)
        self.slow_cell = nn.GRUCell(73, 73)
        self.classifier = nn.Linear(219, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fast = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
=======
        fast = torch.zeros(batch_size, 73, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 73, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 73, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 73, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 73, device=device, dtype=dtype)
>>>>>>> REPLACE