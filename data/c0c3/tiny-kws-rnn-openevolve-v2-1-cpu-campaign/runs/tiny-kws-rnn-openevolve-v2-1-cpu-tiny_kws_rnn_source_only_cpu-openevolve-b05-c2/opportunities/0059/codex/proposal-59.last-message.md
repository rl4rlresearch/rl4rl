MECHANISM: Orthonormal centered-logit parameterization

HYPOTHESIS: A symmetric seven-contrast classifier on the qualified 63-unit, 21-step recurrent path will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 269,920,665.

INTENDED_EDIT: Restore the qualified 63-unit GRU and 21-frame schedule, then replace eight directly learned logits with seven learned orthonormal contrasts expanded into eight centered logits using fixed arithmetic.

EVIDENCE: The 63-unit, 21-step model qualified at 85.03%. Fixing one class logit at zero reduced the target MACs but failed at 82.70%; an orthonormal centered basis preserves the same seven-dimensional softmax space without privileging a reference class or introducing its asymmetric conditioning.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 63, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 7)
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
        return self.classifier(
            torch.cat(
                (mean_output, final_output, maximum, minimum),
                dim=-1,
            )
        )
=======
        contrasts = self.classifier(
            torch.cat(
                (mean_output, final_output, maximum, minimum),
                dim=-1,
            )
        )
        inv_sqrt_eight = 1.0 / math.sqrt(8.0)
        basis_norm = math.sqrt(2.0 - 2.0 * inv_sqrt_eight)
        first_coefficient = -inv_sqrt_eight / basis_norm
        last_coefficient = (1.0 - inv_sqrt_eight) / basis_norm
        projection = first_coefficient * contrasts.sum(dim=-1, keepdim=True)
        return torch.cat(
            (
                contrasts - 2.0 * first_coefficient * projection,
                -2.0 * last_coefficient * projection,
            ),
            dim=-1,
        )
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