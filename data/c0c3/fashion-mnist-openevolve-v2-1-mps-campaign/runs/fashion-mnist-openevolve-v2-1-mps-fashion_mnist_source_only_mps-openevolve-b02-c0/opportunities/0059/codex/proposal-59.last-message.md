MECHANISM: Batch-normalization folding for fast calibrated ten-view inference

HYPOTHESIS: Folding evaluation-time batch normalization into adjacent convolutions and the classifier projection will prevent verification timeout while preserving 9,252 predictions, and a 1.20 ensemble scale will reduce cross-entropy below 0.2235582.

INTENDED_EDIT: Fuse batch-normalization parameters into cached inference-only weights, retain the complete ten-view ensemble, and raise its calibration scale from 1.10 to 1.20.

EVIDENCE: The 1.10 ten-view design completed with 9,252 correct, while scaling from 1.05 to 1.10 lowered cross-entropy without changing predictions; subsequent equally expensive inference variants timed out, motivating an exactly equivalent but cheaper evaluation path.

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def train(self, mode: bool = True) -> ImageClassifier:
        if mode:
            self._eval_fused_layers = None
        return super().train(mode)

    def _get_eval_fused_layers(self):
        cached = getattr(self, "_eval_fused_layers", None)
        if cached is not None:
            return cached

        fused_layers = []
        with torch.no_grad():
            for convolution_index, normalization_index in (
                (0, 1),
                (3, 4),
                (8, 9),
                (11, 12),
            ):
                convolution = self.features[convolution_index]
                normalization = self.features[normalization_index]
                scale = normalization.weight.detach() * torch.rsqrt(
                    normalization.running_var.detach() + normalization.eps
                )
                base_bias = (
                    convolution.bias.detach()
                    if convolution.bias is not None
                    else torch.zeros_like(normalization.running_mean)
                )
                fused_layers.append(
                    (
                        convolution.weight.detach()
                        * scale[:, None, None, None],
                        normalization.bias.detach()
                        + (base_bias - normalization.running_mean.detach()) * scale,
                    )
                )

            projection = self.classifier[1]
            normalization = self.classifier[2]
            scale = normalization.weight.detach() * torch.rsqrt(
                normalization.running_var.detach() + normalization.eps
            )
            fused_layers.append(
                (
                    projection.weight.detach() * scale[:, None],
                    normalization.bias.detach()
                    + (projection.bias.detach() - normalization.running_mean.detach())
                    * scale,
                )
            )

        self._eval_fused_layers = tuple(fused_layers)
        return self._eval_fused_layers

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.classifier(self.features(images))

        conv1, conv2, conv3, conv4, projection = (
            self._get_eval_fused_layers()
        )
        features = F.relu(
            F.conv2d(images, conv1[0], conv1[1], padding=1), inplace=True
        )
        features = F.relu(
            F.conv2d(features, conv2[0], conv2[1], padding=1), inplace=True
        )
        features = F.max_pool2d(features, 2)
        features = F.relu(
            F.conv2d(features, conv3[0], conv3[1], padding=1), inplace=True
        )
        features = F.relu(
            F.conv2d(features, conv4[0], conv4[1], padding=1), inplace=True
        )
        features = F.max_pool2d(features, 2)
        features = torch.flatten(features, 1)
        features = F.relu(
            F.linear(features, projection[0], projection[1]), inplace=True
        )
        return self.classifier[5](features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.10 * ensemble_log_probabilities
=======
        return 1.20 * ensemble_log_probabilities
>>>>>>> REPLACE