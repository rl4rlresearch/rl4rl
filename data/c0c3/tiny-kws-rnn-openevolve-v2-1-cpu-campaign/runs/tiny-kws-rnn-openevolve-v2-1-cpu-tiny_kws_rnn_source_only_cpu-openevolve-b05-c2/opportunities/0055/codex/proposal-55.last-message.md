MECHANISM: Fixed edge-band pooling with qualified recurrent width

HYPOTHESIS: Averaging the two highest mel bands into one feature before the qualified 63-unit, 21-step GRU will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 266,891,310.

INTENDED_EDIT: Restore the qualified 63-unit and 21-frame design, then reduce its recurrent input width from 20 to 19 using fixed pooling of the final two normalized mel bands.

EVIDENCE: The 63-unit, 21-step GRU qualified at 85.03%, whereas the more aggressive learned 20-to-16 bottleneck failed at 83.68%; a one-feature fixed reduction preserves information from every mel band while testing the smallest spectral contraction.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(19, 63, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 63, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 63, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, maximum, minimum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
=======
        hidden, summary, maximum, minimum, count = state
        normalized = self.input_norm(frame)
        compressed = torch.cat(
            (normalized[..., :18], normalized[..., 18:].mean(dim=-1, keepdim=True)),
            dim=-1,
        )
        output, hidden = self.gru(
            compressed.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, maximum, minimum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
=======
        hidden, summary, maximum, minimum, count = state
        normalized = self.input_norm(frames)
        compressed = torch.cat(
            (normalized[..., :18], normalized[..., 18:].mean(dim=-1, keepdim=True)),
            dim=-1,
        )
        outputs, hidden = self.gru(
            compressed, hidden.transpose(0, 1).contiguous()
        )
        return (
>>>>>>> REPLACE

<<<<<<< SEARCH
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
=======
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
>>>>>>> REPLACE