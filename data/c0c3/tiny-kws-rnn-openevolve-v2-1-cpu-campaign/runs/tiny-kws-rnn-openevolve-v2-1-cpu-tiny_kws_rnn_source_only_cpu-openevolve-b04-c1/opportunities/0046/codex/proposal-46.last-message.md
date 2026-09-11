MECHANISM: Temporal-pyramid readout for recurrent-width pruning

HYPOTHESIS: A 67-unit paired-frame GRU with the successful six-transition early pool split into two three-transition summaries will retain at least 85% accuracy while reducing total inference MACs from 234,980,800 to approximately 229,777,840.

INTENDED_EDIT: Reduce GRU width from 68 to 67 and replace the early/late/max/final readout with first-early/second-early/late/max/final pooling, preserving the established transition-six boundary while adding inexpensive temporal resolution.

EVIDENCE: The 68-unit four-view model achieved 85.77%, and at 69 units the informative final-state view improved accuracy from 85.28% to 85.89%. This shows that cheap temporally structured readout capacity can support recurrent-width pruning.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(272, 7)
=======
        self.gru = nn.GRU(40, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(335, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        middle_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
>>>>>>> REPLACE

<<<<<<< SEARCH
            return (
                hidden,
                early_summary,
                late_summary,
                maximum,
                count,
                frame,
                torch.ones_like(phase),
            )
=======
            return (
                hidden,
                early_summary,
                middle_summary,
                late_summary,
                maximum,
                count,
                frame,
                torch.ones_like(phase),
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        early_weight = (count < 6.0).to(dtype=output.dtype)
        late_weight = 1.0 - early_weight
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            late_summary + late_weight * output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
        )
=======
        early_weight = (count < 3.0).to(dtype=output.dtype)
        middle_weight = (
            (count >= 3.0) & (count < 6.0)
        ).to(dtype=output.dtype)
        late_weight = (count >= 6.0).to(dtype=output.dtype)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            middle_summary + middle_weight * output,
            late_summary + late_weight * output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
>>>>>>> REPLACE

<<<<<<< SEARCH
            early_weights = (positions < 6.0).to(dtype=outputs.dtype)
            late_weights = 1.0 - early_weights
            early_summary = early_summary + (
                outputs * early_weights
            ).sum(dim=1)
            late_summary = late_summary + (
                outputs * late_weights
            ).sum(dim=1)
=======
            early_weights = (positions < 3.0).to(dtype=outputs.dtype)
            middle_weights = (
                (positions >= 3.0) & (positions < 6.0)
            ).to(dtype=outputs.dtype)
            late_weights = (positions >= 6.0).to(dtype=outputs.dtype)
            early_summary = early_summary + (
                outputs * early_weights
            ).sum(dim=1)
            middle_summary = middle_summary + (
                outputs * middle_weights
            ).sum(dim=1)
            late_summary = late_summary + (
                outputs * late_weights
            ).sum(dim=1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )

    def classify(
=======
        return (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )

    def classify(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        del pending, phase
        early_count = count.clamp(max=6.0).clamp_min(1.0)
        late_count = (count - 6.0).clamp_min(1.0)
        pooled = torch.cat(
            (
                early_summary / early_count,
                late_summary / late_count,
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
=======
    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        del pending, phase
        early_count = count.clamp(max=3.0).clamp_min(1.0)
        middle_count = (count - 3.0).clamp(max=3.0).clamp_min(1.0)
        late_count = (count - 6.0).clamp_min(1.0)
        pooled = torch.cat(
            (
                early_summary / early_count,
                middle_summary / middle_count,
                late_summary / late_count,
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
>>>>>>> REPLACE