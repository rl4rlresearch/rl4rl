MECHANISM: Pyramidal two-frame upper recurrence

HYPOTHESIS: Updating the upper GRU from averaged adjacent acoustic frames and every second lower-GRU output will retain at least 85% validation accuracy while reducing total inference MACs from 600.70M to approximately 415.04M.

INTENDED_EDIT: Keep the load-bearing lower GRU at all 28 causal steps, but clock the upper GRU once per frame pair, add pending-acoustic and upper-update-count state, and make sequence execution exactly match repeated recurrent steps.

EVIDENCE: The 49+48 hierarchy reached 86.01%, upper-skip pruning and folding repeatedly retained at least 85%, and reducing the lower width failed; this challenges the shared assumption that the comparatively redundant upper stage must execute at every frame while preserving full-rate lower recurrence.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A raw-skip hierarchy of two compact causal recurrent stages."""
=======
class KeywordGRU(nn.Module):
    """A pyramidal hierarchy with a full-rate lower and half-rate upper stage."""
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
    ]:
        lower_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return lower_hidden, upper_hidden, lower_summary, upper_summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        lower_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        upper_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending_acoustic = torch.zeros(
            batch_size, 16, device=device, dtype=dtype
        )
        return (
            lower_hidden,
            upper_hidden,
            lower_summary,
            upper_summary,
            count,
            upper_count,
            pending_acoustic,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        normalized = self.input_norm(self._fold_bands(frame))
        lower_output, lower_hidden = self.lower_gru(
            normalized.unsqueeze(1),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :14],
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
        upper_output, upper_hidden = self.upper_gru(
            upper_input.unsqueeze(1),
            upper_hidden.transpose(0, 1).contiguous(),
        )
        upper_output = upper_output[:, 0, :]
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            lower_summary + lower_output,
            upper_summary + upper_output,
            count + 1.0,
        )
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            lower_hidden,
            upper_hidden,
            lower_summary,
            upper_summary,
            count,
            upper_count,
            pending_acoustic,
        ) = state
        normalized = self.input_norm(self._fold_bands(frame))
        lower_output, lower_hidden_time = self.lower_gru(
            normalized.unsqueeze(1),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        lower_output = lower_output[:, 0, :]
        lower_hidden = lower_hidden_time.transpose(0, 1)
        upper_acoustic = torch.cat(
            (
                normalized[..., :14],
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )

        if int(count[0, 0].item()) % 2 == 0:
            pending_acoustic = upper_acoustic
        else:
            pair_acoustic = 0.5 * (pending_acoustic + upper_acoustic)
            upper_input = torch.cat((pair_acoustic, lower_output), dim=-1)
            upper_output, upper_hidden_time = self.upper_gru(
                upper_input.unsqueeze(1),
                upper_hidden.transpose(0, 1).contiguous(),
            )
            upper_hidden = upper_hidden_time.transpose(0, 1)
            upper_summary = upper_summary + upper_output[:, 0, :]
            upper_count = upper_count + 1.0
            pending_acoustic = torch.zeros_like(pending_acoustic)

        return (
            lower_hidden,
            upper_hidden,
            lower_summary + lower_output,
            upper_summary,
            count + 1.0,
            upper_count,
            pending_acoustic,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        normalized = self.input_norm(self._fold_bands(frames))
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                normalized[..., :14],
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
        upper_outputs, upper_hidden = self.upper_gru(
            upper_inputs,
            upper_hidden.transpose(0, 1).contiguous(),
        )
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            lower_summary + lower_outputs.sum(dim=1),
            upper_summary + upper_outputs.sum(dim=1),
            count + frames.shape[1],
        )
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            lower_hidden,
            upper_hidden,
            lower_summary,
            upper_summary,
            count,
            upper_count,
            pending_acoustic,
        ) = state
        normalized = self.input_norm(self._fold_bands(frames))
        lower_outputs, lower_hidden_time = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        lower_hidden = lower_hidden_time.transpose(0, 1)
        upper_acoustic = torch.cat(
            (
                normalized[..., :14],
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )

        total_steps = frames.shape[1]
        if int(count[0, 0].item()) % 2 == 0:
            paired_steps = (total_steps // 2) * 2
            pair_acoustic = 0.5 * (
                upper_acoustic[:, 0:paired_steps:2, :]
                + upper_acoustic[:, 1:paired_steps:2, :]
            )
            pair_lower = lower_outputs[:, 1:paired_steps:2, :]
            if total_steps % 2:
                pending_acoustic = upper_acoustic[:, -1, :]
            else:
                pending_acoustic = torch.zeros_like(pending_acoustic)
        else:
            first_acoustic = 0.5 * (
                pending_acoustic + upper_acoustic[:, 0, :]
            )
            pair_acoustic = first_acoustic.unsqueeze(1)
            pair_lower = lower_outputs[:, 0:1, :]
            paired_steps = ((total_steps - 1) // 2) * 2
            if paired_steps:
                remaining_acoustic = 0.5 * (
                    upper_acoustic[:, 1 : 1 + paired_steps : 2, :]
                    + upper_acoustic[:, 2 : 1 + paired_steps : 2, :]
                )
                pair_acoustic = torch.cat(
                    (pair_acoustic, remaining_acoustic), dim=1
                )
                pair_lower = torch.cat(
                    (
                        pair_lower,
                        lower_outputs[:, 2 : 1 + paired_steps : 2, :],
                    ),
                    dim=1,
                )
            if (total_steps - 1) % 2:
                pending_acoustic = upper_acoustic[:, -1, :]
            else:
                pending_acoustic = torch.zeros_like(pending_acoustic)

        if pair_acoustic.shape[1]:
            upper_inputs = torch.cat((pair_acoustic, pair_lower), dim=-1)
            upper_outputs, upper_hidden_time = self.upper_gru(
                upper_inputs,
                upper_hidden.transpose(0, 1).contiguous(),
            )
            upper_hidden = upper_hidden_time.transpose(0, 1)
            upper_summary = upper_summary + upper_outputs.sum(dim=1)
            upper_count = upper_count + pair_acoustic.shape[1]

        return (
            lower_hidden,
            upper_hidden,
            lower_summary + lower_outputs.sum(dim=1),
            upper_summary,
            count + total_steps,
            upper_count,
            pending_acoustic,
        )
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
        ],
    ) -> torch.Tensor:
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        divisor = count.clamp_min(1.0)
        features = torch.cat(
            (
                lower_summary / divisor,
                upper_summary / divisor,
                lower_hidden[:, 0, :],
                upper_hidden[:, 0, :],
            ),
            dim=-1,
        )
        return self.classifier(features)
=======
    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (
            lower_hidden,
            upper_hidden,
            lower_summary,
            upper_summary,
            count,
            upper_count,
            pending_acoustic,
        ) = state
        del pending_acoustic
        features = torch.cat(
            (
                lower_summary / count.clamp_min(1.0),
                upper_summary / upper_count.clamp_min(1.0),
                lower_hidden[:, 0, :],
                upper_hidden[:, 0, :],
            ),
            dim=-1,
        )
        return self.classifier(features)
>>>>>>> REPLACE