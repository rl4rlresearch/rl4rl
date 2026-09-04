MECHANISM: Fast approximate-GELU top-four channel saliency

HYPOTHESIS: Restoring top-four saliency and plain paired-view cross-entropy while using tanh-approximated GELUs will exceed 9,322 correct predictions and finish within the verification limit by preserving the strongest verified model while reducing activation overhead.

INTENDED_EDIT: Replace maximum saliency with top-four averaging, remove the harmful ensemble-aware loss, and use tanh-approximated GELU throughout the network.

EVIDENCE: Top-four saliency with plain paired-view cross-entropy achieved the best verified result of 9,322 correct, but later top-four trials timed out; hard-maximum attention finished faster but peaked at 9,320, motivating a compute-focused restoration of the strongest design.

<<<<<<< SEARCH
from torch.nn import functional as F

BATCH_SIZE = 128
=======
from torch.nn import functional as F


def _fast_gelu(inputs: torch.Tensor) -> torch.Tensor:
    return F.gelu(inputs, approximate="tanh")


BATCH_SIZE = 128
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
=======
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(approximate="tanh"),
            nn.Dropout(p=0.10),
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = F.gelu(self.stem_norm(self.stem_conv(images)))
        residual = features
        features = F.gelu(
            self.stem_refine_norm1(self.stem_refine_conv1(features))
        )
        features = self.stem_refine_norm2(self.stem_refine_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = self.projection(features)
        features = F.gelu(self.block_norm1(self.block_conv1(features)))
        features = self.block_norm2(self.block_conv2(features))
        features = self.pool(F.gelu(features + residual))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = _fast_gelu(self.stem_norm(self.stem_conv(images)))
        residual = features
        features = _fast_gelu(
            self.stem_refine_norm1(self.stem_refine_conv1(features))
        )
        features = self.stem_refine_norm2(self.stem_refine_conv2(features))
        features = self.pool(_fast_gelu(features + residual))

        residual = self.projection(features)
        features = _fast_gelu(self.block_norm1(self.block_conv1(features)))
        features = self.block_norm2(self.block_conv2(features))
        features = self.pool(_fast_gelu(features + residual))
>>>>>>> REPLACE

<<<<<<< SEARCH
        residual = features
        features = F.gelu(self.deep_norm1(self.deep_conv1(features)))
        features = F.gelu(self.deep_norm2(self.deep_conv2(features)))
        features = self.deep_norm3(self.deep_conv3(features))
        features = F.gelu(features + residual)

        residual = features
        features = F.gelu(
            self.post_deep_norm1(self.post_deep_conv1(features))
        )
        features = F.gelu(
            self.post_deep_norm2(self.post_deep_conv2(features))
        )
        features = self.post_deep_norm3(self.post_deep_conv3(features))
        features = F.gelu(features + residual)
=======
        residual = features
        features = _fast_gelu(self.deep_norm1(self.deep_conv1(features)))
        features = _fast_gelu(self.deep_norm2(self.deep_conv2(features)))
        features = self.deep_norm3(self.deep_conv3(features))
        features = _fast_gelu(features + residual)

        residual = features
        features = _fast_gelu(
            self.post_deep_norm1(self.post_deep_conv1(features))
        )
        features = _fast_gelu(
            self.post_deep_norm2(self.post_deep_conv2(features))
        )
        features = self.post_deep_norm3(self.post_deep_conv3(features))
        features = _fast_gelu(features + residual)
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    individual_loss = F.cross_entropy(paired_logits, paired_labels)

    batch_size = images.shape[0]
    ensemble_log_probabilities = torch.logsumexp(
        torch.stack(
            (
                F.log_softmax(paired_logits[:batch_size], dim=1),
                F.log_softmax(paired_logits[batch_size:], dim=1),
            )
        ),
        dim=0,
    ) - math.log(2.0)
    ensemble_loss = F.nll_loss(ensemble_log_probabilities, labels)
    return 0.5 * (individual_loss + ensemble_loss)
=======
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
>>>>>>> REPLACE