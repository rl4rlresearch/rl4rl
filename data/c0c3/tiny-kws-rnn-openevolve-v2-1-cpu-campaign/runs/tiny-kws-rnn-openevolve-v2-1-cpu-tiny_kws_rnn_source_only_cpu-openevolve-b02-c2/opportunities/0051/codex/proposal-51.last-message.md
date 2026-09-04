MECHANISM: Endpoint-preserving adjacent mel-band fusion

HYPOTHESIS: Fusing two central adjacent mel bands will preserve at least 85% accuracy on the qualified 25-step schedule while reducing total inference MACs from 519,679,860 to approximately 514,545,360.

INTENDED_EDIT: Use frames 4–28 and compress bands 1–17 into 16 recurrent features by averaging bands 8 and 9 rather than discarding any band.

EVIDENCE: Bands 1–17 qualified, while removing either informative spectral endpoint failed; adjacent fusion tests whether correlated interior bands can share one feature without losing the endpoint evidence.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
        self.input_norm = nn.LayerNorm(16)
        self.gru = nn.GRU(16, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)

    @staticmethod
    def spectral_features(frames: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                frames[..., 1:8],
                frames[..., 8:10].mean(dim=-1, keepdim=True),
                frames[..., 10:18],
            ),
            dim=-1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame[..., 1:18]).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        output, hidden = self.gru(
            self.input_norm(self.spectral_features(frame)).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames[..., 1:18]), hidden.transpose(0, 1).contiguous()
        )
=======
        outputs, hidden = self.gru(
            self.input_norm(self.spectral_features(frames)),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(4, available_frames - 3))
>>>>>>> REPLACE