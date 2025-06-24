import torch
import torch.nn.functional as F

def attention_consistency_loss(final_embedding, fc_weights, labels):
    """
    Enforces consistency in attention scores among patients with the same disease.

    final_embedding: Tensor of shape (B, N) -> Node embeddings for each patient
    fc_weights: Tensor of shape (num_classes, N, 1) -> Final fc weights for prediction
    labels: Tensor of shape (B,) -> disease labels for each patient

    Returns:
    consistency_loss: Scalar loss value.
    """
    unique_labels = labels.unique()  # Get distinct disease labels
    consistency_loss = 0.0
    count = 0

    for label in unique_labels:
        if label == 0:
            continue
        mask = (labels == label)  # Select patients with this disease
        if mask.sum() < 2:  # Skip if there's only one sample
            continue
        
        group_embed = final_embedding[mask].squeeze()  # Select attention scores for this disease (shape: [B_class, N])
        # print('group_embed shape', group_embed.shape)
        # Get fc weight, shape [N]
        fc_w = fc_weights[int(label)].squeeze(-1)
        # print('fc w shape', fc_w.shape)

        # Inner product, [B_class, 1
        # print('group embed, fc_w', group_embed.shape, fc_w.unsqueeze(-1).shape)
        group_scores = torch.matmul(group_embed, fc_w.unsqueeze(-1))
        # print('group scores', group_scores.shape)
        
        # Compute the mean score for this group, shape: [1, 1]
        mean_score = group_scores.mean(dim=0, keepdim=True)

        # Expand the mean to the same shape as group_scores
        mean_score_expanded = mean_score.expand_as(group_scores)

        # Compute MSE loss between each sample's score and the group mean
        group_loss = F.mse_loss(group_scores, mean_score_expanded)
        consistency_loss += group_loss
        count += 1
        
    return consistency_loss / max(count, 1)  # Normalize by number of classes


def final_embedding_entropy_loss(embeddings, fc_weights, labels):
    """
    Computes an entropy-based sparsity loss on the final embeddings.

    This version does the following:
      - Extracts embeddings for non-normal cases (labels != 0).
      - Uses the fc layer weights for non-normal classes. Here, fc_weights is given as [4, num_rois, 1],
        where index 0 corresponds to the normal class. We extract indices 1: (non-normal), then reshape them
        to [1, num_rois, 3].
      - Performs an elementwise multiplication between each ROI's embedding (shape: [batch, num_rois, 1])
        and the corresponding fc weight to produce scores for each ROI and each of the 3 non-normal classes
        (resulting shape: [batch, num_rois, 3]).
      - Computes the entropy of these scores (after a log-softmax) and averages over ROIs and batch.

    Args:
        embeddings (torch.Tensor): Final embeddings of shape (batch, num_rois, 1).
        fc_weights (torch.Tensor): Fully connected layer weights with shape (4, num_rois, 1) where index 0
                                   corresponds to the normal class.
        labels (torch.Tensor): Labels for each sample of shape (batch,). A value of 0 indicates normal.
        
    Returns:
        torch.Tensor: A scalar representing the average entropy loss.
    """
    # Select only the embeddings corresponding to non-normal cases.
    mask = (labels != 0)
    # If no non-normal sample exists, we return zero loss.
    if mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Selected embeddings have shape (N, num_rois, 1) where N is the number of non-normal samples.
    selected_embeddings = embeddings[mask]
    
    fc_non_normal = fc_weights[1:]  # shape: (3, num_rois, 1)
    fc_non_normal = fc_non_normal.squeeze(-1).transpose(0, 1) # (num_rois, 3)
    fc_w = fc_non_normal.unsqueeze(0)
    
    # Compute the inner product between each ROI embedding (a scalar) and the fc weight vector.
    # Broadcasting: (N, num_rois, 1) * (1, num_rois, 3) => (N, num_rois, 3)
    # print(selected_embeddings.shape, fc_w.shape)
    prod = selected_embeddings * fc_w
    
    # Compute the log-softmax over the last dimension (3 classes) for numerical stability.
    log_p = F.log_softmax(prod, dim=2)
    # Recover probabilities and add a small constant for stability.
    p = torch.exp(log_p) + 1e-6
    
    # Compute the entropy per ROI and then average over the ROIs and samples.
    # This yields a scalar loss value.
    entropy = -torch.sum(p * log_p, dim=2).mean()
    
    return entropy

