MECHANISM: Max-pooled candidate-biased gated-state width trim

HYPOTHESIS: A 97-unit state with 31/32/33 reset-update-candidate feedback and mean-max-endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs to approximately 320,647,080.

INTENDED_EDIT: Replace the current 101-unit recurrence with a 97-unit candidate-biased bottleneck recurrence and add elementwise temporal maximum pooling to the classifier state.

EVIDENCE: The 98-unit 31/32/33 model with mean-max-endpoint readout achieved 86.871% accuracy at 323,952,720 MACs, providing enough observed margin to motivate the nearest one-unit state reduction.

<<<<<<< SEARCH
        self.input_projection = nn.Linear(20, 303)
        self.recurrent_down = nn.Linear(101, 96)
        self.recurrent_up = nn.ModuleList(
            [nn.Linear(32, 101) for _ in range(3)]
        )
        self.classifier = nn.Linear(202, 8)
=======
        self.input_projection = nn.Linear(20, 291)
        self.recurrent_down = nn.Linear(97, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 97),
                nn.Linear(32, 97),
                nn.Linear(33, 97),
            ]
        )
        self.classifier = nn.Linear(291, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 101, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 101, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 97), -1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count
>>>>>>> REPLACE

<<<<<<< SEARCH
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
=======
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
>>>>>>> REPLACE

<<<<<<< SEARCH
        reset_code, update_code, candidate_code = torch.tanh(
            self.recurrent_down(hidden)
        ).chunk(3, dim=1)
=======
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 33), dim=1
        )
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