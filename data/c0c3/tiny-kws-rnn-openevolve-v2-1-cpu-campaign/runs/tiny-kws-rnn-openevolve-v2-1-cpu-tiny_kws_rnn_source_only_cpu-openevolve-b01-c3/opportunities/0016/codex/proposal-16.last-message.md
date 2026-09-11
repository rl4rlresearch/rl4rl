MECHANISM: One-unit recurrent width boundary test

HYPOTHESIS: A 104-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 514,134,600 to approximately 505,169,600.

INTENDED_EDIT: Reduce the recurrent state, temporal summary, and classifier width from 105 to 104 units while preserving the qualified batch-64 training regime and 16-frame schedule.

EVIDENCE: The adjacent 105-unit design achieved 85.644% accuracy at 514,134,600 MACs, leaving a 0.644-point margin and making a one-unit reduction the smallest informative test below the current cost boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 105, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(105, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 105, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 105, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
>>>>>>> REPLACE