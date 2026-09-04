MECHANISM: Boundary-width GRU pruning with triple temporal readout

HYPOTHESIS: A 73-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.4% versus the qualified 74-unit design.

INTENDED_EDIT: Reduce the GRU and recurrent-summary width from 76 to 73 units and resize the triple-readout classifier from 228 to 219 inputs, preserving all 32 causal steps and the established training procedure.

EVIDENCE: The 74-unit design achieved 87.85% validation accuracy—the strongest result among the tested 74–80-unit triple-readout models—with a 2.85-point margin above the requirement, motivating a one-unit structural reduction to probe the next compute boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 76, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(228, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 76, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 73, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(219, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 73, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 73, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 73, device=device, dtype=dtype)
>>>>>>> REPLACE