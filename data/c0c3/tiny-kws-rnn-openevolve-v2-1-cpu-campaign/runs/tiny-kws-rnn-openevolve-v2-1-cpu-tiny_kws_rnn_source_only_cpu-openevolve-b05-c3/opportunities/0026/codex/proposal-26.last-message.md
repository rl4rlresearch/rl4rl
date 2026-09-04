MECHANISM: Adjacent width-boundary reduction of the dual-readout GRU

HYPOTHESIS: A 115-unit GRU with the qualified 20-frame mean/endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 772,959,040 to approximately 760,672,100.

INTENDED_EDIT: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 118 to 115 units while preserving the qualified schedule and training procedure.

EVIDENCE: The adjacent 116-unit design achieved 85.77% accuracy at 772,959,040 MACs, and every tested width from 116 through 126 qualified; testing 115 units is the most direct remaining cost-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 118, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(118, 8)
        self.endpoint_classifier = nn.Linear(118, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 118, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 118, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 115, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(115, 8)
        self.endpoint_classifier = nn.Linear(115, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 115, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 115, device=device, dtype=dtype)
>>>>>>> REPLACE