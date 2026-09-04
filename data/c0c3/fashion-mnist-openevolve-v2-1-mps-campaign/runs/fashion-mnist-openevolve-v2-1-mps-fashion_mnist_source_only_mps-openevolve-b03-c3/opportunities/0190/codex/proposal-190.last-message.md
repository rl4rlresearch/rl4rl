MECHANISM: Low-rank nonlocal spatial-relation residual

HYPOTHESIS: Adding content-dependent pairwise interactions across the 7×7 feature map will exceed 9,284 correct predictions by modeling relationships between distant image parts that the local refinement stack cannot represent directly.

INTENDED_EDIT: Preserve the validated convolutional, aggregation, classifier, training, and TTA paths while adding a 3,841-parameter low-rank spatial-attention residual before aggregation; total learned parameters become 248,993.

EVIDENCE: Mean–max channel attention fell to 9,274 correct, indicating adaptive channel importance was not the missing mechanism. This patch instead challenges the shared assumption that local convolutions alone sufficiently relate separated shape parts, while retaining the established layout-sensitive decision path whose replacement previously regressed.

<<<<<<< SEARCH
class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weight * maximum + (1.0 - max_weight) * average


class ImageClassifier(nn.Module):
=======
class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weight * maximum + (1.0 - max_weight) * average


class SpatialRelation2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Conv2d(80, 8, kernel_size=1, bias=False)
        self.key = nn.Conv2d(80, 8, kernel_size=1, bias=False)
        self.value = nn.Conv2d(80, 16, kernel_size=1, bias=False)
        self.project = nn.Conv2d(16, 80, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        queries = self.query(features).flatten(2).transpose(1, 2)
        keys = self.key(features).flatten(2)
        attention = torch.softmax(
            torch.bmm(queries, keys) * (8.0 ** -0.5), dim=-1
        )
        values = self.value(features).flatten(2).transpose(1, 2)
        context = torch.bmm(attention, values).transpose(1, 2)
        context = context.reshape(batch, 16, height, width)
        return self.scale * self.project(context)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.aggregate_local = nn.Sequential(
=======
        self.relate = SpatialRelation2d()
        self.aggregate_local = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        features = torch.cat(
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(
            features + self.refine(features) + self.relate(features)
        )
        features = torch.cat(
>>>>>>> REPLACE