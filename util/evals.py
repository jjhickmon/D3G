import torch

def get_all_accuracies(predictions, true_labels, classes, device):
    all_class_labels = torch.arange(start=0, end=len(classes))
    pred_masks = predictions.unsqueeze(0) == all_class_labels.unsqueeze(1).to(device)
    true_masks = true_labels.unsqueeze(0) == all_class_labels.unsqueeze(1).to(device)
    # the extra precision is not necessary but to ensure accuracy is not lost
    per_class_acc = ((true_masks & pred_masks).double().sum(dim=1) / true_masks.double().sum(dim=1)) * 100
    avg_per_class_acc = per_class_acc.mean(dtype=torch.double).item()
    labeled_per_class_acc = dict(zip(classes, per_class_acc.tolist()))
    total_acc = (predictions == true_labels).mean(dtype=torch.double).item() * 100
    return (total_acc, avg_per_class_acc, labeled_per_class_acc)

def equigen_weighted_sum_embeddings(true_embeddings, gen_text_embeddings, gen_img_embeddings, text_weight, gen_weight, dataset_classnames, batch_size, device):
    all_preds, all_true_labels = [], []
    for true_image_embeddings, true_labels in zip(torch.split(true_embeddings[0], batch_size), torch.split(true_embeddings[1], batch_size)):
        pred_class_embeddings = gen_img_embeddings * gen_weight + gen_text_embeddings * text_weight
        pred_class_embeddings /= pred_class_embeddings.norm(dim=1, keepdim=True)
        similarity_scores = true_image_embeddings @ pred_class_embeddings.T
        all_preds.append(similarity_scores.argmax(dim=1))
        all_true_labels.append(true_labels.to(device))
    return get_all_accuracies(torch.cat(all_preds), torch.cat(all_true_labels), dataset_classnames, device)

def standard_clip_baseline(true_embeddings, gen_text_embeddings, dataset_classnames, batch_size, device):
    all_preds, all_true_labels = [], []
    for true_image_embeddings, true_labels in zip(torch.split(true_embeddings[0], batch_size), torch.split(true_embeddings[1], batch_size)):
        similarity_scores = true_image_embeddings @ gen_text_embeddings.T
        all_preds.append(similarity_scores.argmax(dim=1))
        all_true_labels.append(true_labels.to(device))
    return get_all_accuracies(torch.cat(all_preds), torch.cat(all_true_labels), dataset_classnames, device)