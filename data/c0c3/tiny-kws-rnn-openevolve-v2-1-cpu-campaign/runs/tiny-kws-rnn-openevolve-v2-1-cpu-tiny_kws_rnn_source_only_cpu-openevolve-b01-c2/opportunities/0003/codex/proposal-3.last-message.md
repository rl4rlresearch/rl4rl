MECHANISM: Full-resolution narrow recurrence with dual temporal readout

HYPOTHESIS: A 96-unit GRU processing all 32 frames, with mean and final-state features, will retain at least 85% accuracy while reducing total inference MACs from 1.483B to approximately 0.873B.

INTENDED_EDIT: Reduce GRU width from 128 to 96, preserve the full frame schedule, and classify from concatenated mean-pooled and final recurrent states.

EVIDENCE: Thinning to 24 frames reduced accuracy to 84.17% despite retaining the 128-unit GRU, suggesting temporal observations are more valuable than excess hidden width; width reduction preserves every frame while cutting the quadratic recurrent matrix cost.

<<<<<<< SEARCH
    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 96
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(2 * self.hidden_size, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        return self.classifier(features)
>>>>>>> REPLACE