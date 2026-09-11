MECHANISM: Adjacent-width clocked dual-timescale GRU

HYPOTHESIS: A 75-unit fast GRU over 20 frames plus a 75-unit slow GRU over four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 471,239,520 to approximately 459,904,500.

INTENDED_EDIT: Reduce both recurrent widths to 75 units and resize the three-summary classifier input to 225 features, preserving the qualified schedule and training procedure.

EVIDENCE: The 76-unit design achieved 87.73% validation accuracy at 471,239,520 MACs, leaving a 2.73-point margin; the adjacent 75-unit width is the most informative remaining capacity-boundary probe.

<<<<<<< SEARCH
        self.fast_cell = nn.GRUCell(20, 78)
        self.slow_norm = nn.LayerNorm(78)
        self.slow_cell = nn.GRUCell(78, 78)
        self.classifier = nn.Linear(234, 8)
=======
        self.fast_cell = nn.GRUCell(20, 75)
        self.slow_norm = nn.LayerNorm(75)
        self.slow_cell = nn.GRUCell(75, 75)
        self.classifier = nn.Linear(225, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fast = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 78, device=device, dtype=dtype)
=======
        fast = torch.zeros(batch_size, 75, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 75, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 75, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 75, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 75, device=device, dtype=dtype)
>>>>>>> REPLACE