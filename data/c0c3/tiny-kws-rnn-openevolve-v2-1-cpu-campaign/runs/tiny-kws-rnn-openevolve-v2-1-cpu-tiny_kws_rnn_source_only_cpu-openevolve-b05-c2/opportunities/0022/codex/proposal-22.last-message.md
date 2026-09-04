MECHANISM: Adjacent-width GRU pruning with triple temporal readout

HYPOTHESIS: A 70-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.49% versus the qualified 71-unit design.

INTENDED_EDIT: Reduce the GRU and recurrent-summary width from 73 to 70 units and resize the triple-readout classifier from 219 to 210 inputs, preserving all 32 causal steps and the established training procedure.

EVIDENCE: The 71-unit triple-readout model achieved 87.24% validation accuracy at 506,897,400 total MACs, leaving a 2.24-point margin after every tested width from 71 through 80 qualified; this motivates probing the adjacent lower width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 73, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(219, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 73, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 73, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 73, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 70, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(210, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 70, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 70, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 70, device=device, dtype=dtype)
>>>>>>> REPLACE