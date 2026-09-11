MECHANISM: Mean-max-endpoint recurrent readout

HYPOTHESIS: A 98-unit 31/32/33 gated state with explicit temporal maximum pooling will recover at least 85% validation accuracy while reducing total inference MACs below the qualified 99-unit model, to approximately 323,952,720.

INTENDED_EDIT: Reduce the recurrent state to 98 units and augment its mean/final-state classifier with a learned readout from the elementwise maximum recurrent activation across all 16 steps.

EVIDENCE: The 99-unit 31/32/33 model qualified at 85.399% and 326,612,880 MACs, while the adjacent 98-unit mean/final model fell to 83.681%. This tests whether preserving transient keyword evidence in the readout can recover that boundary loss for only 638,960 MACs above the failed 98-unit design.

<<<<<<< SEARCH
        self.input_projection = nn.Linear(20, 300)
        self.recurrent_down = nn.Linear(100, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 100),
                nn.Linear(32, 100),
                nn.Linear(33, 100),
            ]
        )
        self.classifier = nn.Linear(200, 8)
=======
        self.input_projection = nn.Linear(20, 294)
        self.recurrent_down = nn.Linear(98, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 98),
                nn.Linear(32, 98),
                nn.Linear(33, 98),
            ]
        )
        self.classifier = nn.Linear(294, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 98), -1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        hidden = hidden[:, 0, :]
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        hidden = hidden[:, 0, :]
>>>>>>> REPLACE

<<<<<<< SEARCH
        return hidden.unsqueeze(1), summary + hidden, count + 1.0
=======
        return (
            hidden.unsqueeze(1),
            summary + hidden,
            torch.maximum(maximum, hidden),
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        features = torch.cat(
            (
                summary / count.clamp_min(1.0),
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
        return self.classifier(features)
>>>>>>> REPLACE