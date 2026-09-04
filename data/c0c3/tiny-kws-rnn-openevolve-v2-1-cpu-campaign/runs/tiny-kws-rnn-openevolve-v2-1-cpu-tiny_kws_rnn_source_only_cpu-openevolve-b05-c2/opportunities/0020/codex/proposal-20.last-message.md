MECHANISM: Boundary-width GRU pruning with triple temporal readout

HYPOTHESIS: A 72-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.43% versus the qualified 73-unit design.

INTENDED_EDIT: Reduce the GRU and recurrent-summary width from 77 to 72 units and resize the triple-readout classifier from 231 to 216 inputs, preserving all 32 causal steps and the established training procedure.

EVIDENCE: The 73-unit triple-readout model qualified at 85.89% accuracy and 532,599,240 total MACs; testing the adjacent 72-unit width directly probes the next compute boundary without changing the successful temporal readout.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 77, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(231, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 77, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 77, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 77, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 72, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(216, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 72, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 72, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 72, device=device, dtype=dtype)
>>>>>>> REPLACE