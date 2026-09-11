MECHANISM: Class-conditional mixture-of-experts readout

HYPOTHESIS: Replacing the single linear class readout with three input-gated experts per class will exceed 9,311 correct predictions by learning image-dependent decision surfaces while retaining the proven convolutional representation, TTA, and runtime profile at 249,585 parameters.

INTENDED_EDIT: Replace the 147-unit single-head classifier with a 139-unit normalized hidden layer whose class logits are independently blended from three learned experts using per-image, per-class gates.

EVIDENCE: The normalized dense head and weighted-logit TTA reached the best 9,311 correct, but fixed alternative pooling regressed and full-resolution token interaction timed out. This challenges the remaining load-bearing assumption that one fixed linear readout is sufficient, using conditional computation with negligible additional runtime.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class ClassConditionalMixtureHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 139),
            nn.BatchNorm1d(139),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.experts = nn.Linear(139, 3 * 10)
        self.gates = nn.Linear(139, 3 * 10)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(features)
        expert_logits = self.experts(hidden).view(-1, 3, 10)
        expert_weights = F.softmax(
            self.gates(hidden).view(-1, 3, 10),
            dim=1,
        )
        return (expert_logits * expert_weights).sum(dim=1)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
        )
=======
        self.classifier = ClassConditionalMixtureHead()
>>>>>>> REPLACE