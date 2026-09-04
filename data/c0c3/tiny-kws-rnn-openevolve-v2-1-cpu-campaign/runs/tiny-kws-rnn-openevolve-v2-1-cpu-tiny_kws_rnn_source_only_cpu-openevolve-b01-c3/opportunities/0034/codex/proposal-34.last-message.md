MECHANISM: Gated-state width boundary trim

HYPOTHESIS: Reducing the gated state from 104 to 103 units while retaining all three 32-unit feedback bottlenecks will preserve at least 85% validation accuracy and reduce total inference MACs from 343,108,480 to approximately 339,809,360.

INTENDED_EDIT: Resize the input gate projections, recurrent state, gate expansions, temporal summary, and endpoint-augmented classifier from 104 to 103 units while preserving the qualified training procedure and 16-frame schedule.

EVIDENCE: The qualified 104-state, 32/32/32 bottleneck design achieved 85.521% accuracy; unlike the failed bottleneck-rank trims, this nearest structural reduction preserves the full demonstrated feedback capacity of every gate and would improve upon the 340,396,160-MAC selective-reset design.

<<<<<<< SEARCH
        self.input_projection = nn.Linear(20, 312)
        self.recurrent_down = nn.Linear(104, 96)
        self.recurrent_up = nn.ModuleList(
            [nn.Linear(32, 104) for _ in range(3)]
        )
        self.classifier = nn.Linear(208, 8)
=======
        self.input_projection = nn.Linear(20, 309)
        self.recurrent_down = nn.Linear(103, 96)
        self.recurrent_up = nn.ModuleList(
            [nn.Linear(32, 103) for _ in range(3)]
        )
        self.classifier = nn.Linear(206, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
>>>>>>> REPLACE