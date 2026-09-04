MECHANISM: Lightweight flip-logit consistency on reliable maximum attention

HYPOTHESIS: Restoring the 9,320-correct hard-maximum attention and adding a mild centered-logit agreement penalty will exceed 9,322 correct predictions by improving orientation invariance without the harmful ensemble-aware objective or top-k runtime overhead.

INTENDED_EDIT: Replace top-four channel saliency with global-maximum evidence and regularize the already-computed paired-orientation logits toward agreement while retaining ordinary cross-entropy.

EVIDENCE: Hard-maximum attention reached 9,320 correct with lower cross-entropy and faster training than the 9,322-correct top-four model; paired-view training and flip ensembling were beneficial, whereas directly optimizing ensemble likelihood reduced accuracy to 9,307.

<<<<<<< SEARCH
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
=======
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    classification_loss = F.cross_entropy(paired_logits, paired_labels)

    batch_size = images.shape[0]
    first_logits = paired_logits[:batch_size]
    second_logits = paired_logits[batch_size:]
    first_logits = first_logits - first_logits.mean(dim=1, keepdim=True)
    second_logits = second_logits - second_logits.mean(dim=1, keepdim=True)
    consistency_loss = F.mse_loss(first_logits, second_logits)
    return classification_loss + 0.01 * consistency_loss
>>>>>>> REPLACE