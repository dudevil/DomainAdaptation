import numpy as np
import torch
import configs.dann_config as dann_config


# call loss_DANN instead of this function
def _loss_DANN(
        class_predictions_logits,
        logprobs_target,
        instances_labels,
        is_target,
        domain_loss_weight,
        prediction_loss_weight,
        unk_value=dann_config.UNK_VALUE,
        device=torch.device('cpu')
):
    """
    :param class_predictions_logits: Tensor, shape = (batch_size, n_classes).
        Raw (NO logsoftmax).
    :param logprobs_target: Tensor, shape = (batch_size,):
        logprobs that domain is target.
    :param instances_labels: np.Array, shape = (batch_size,)
    :param is_target: np.Array, shape = (batch_size,)
    :param domain_loss_weight: weight of domain loss
    :param prediction_loss_weight: weight of prediction loss
    :param unk_value: value that means that true label is unknown
    """
    instances_labels = instances_labels.long()
    is_target = is_target.float()

    crossentropy = torch.nn.CrossEntropyLoss(ignore_index=unk_value)
    prediction_loss = crossentropy(class_predictions_logits, instances_labels)
    binary_crossentropy = torch.nn.BCEWithLogitsLoss()
    domain_loss = binary_crossentropy(logprobs_target, is_target)
    loss = domain_loss_weight * domain_loss \
           + prediction_loss_weight * prediction_loss
    return loss


# call loss_DANN instead of this function
def _loss_DANN_splitted(
        class_logits_on_src,
        class_logits_on_trg,
        logprobs_target_on_src,
        logprobs_target_on_trg,
        true_labels_on_src,
        true_labels_on_trg,
        domain_loss_weight,
        prediction_loss_weight,
        unk_value=dann_config.UNK_VALUE,
        device=torch.device('cpu'),
):
    """
    :param class_logits_on_src: Tensor, shape = (batch_size, n_classes).
    :param class_logits_on_trg: Tensor, shape = (batch_size, n_classes).
    :param logprobs_target_on_src: Tensor, shape = (batch_size,):
    :param logprobs_target_on_trg: Tensor, shape = (batch_size,):
    :param true_labels_on_src: np.Array, shape = (batch_size,)
    :param true_labels_on_trg: np.Array, shape = (batch_size,)
    :param domain_loss_weight: weight of domain loss
    :param prediction_loss_weight: weight of prediction loss
    :param unk_value: value that means that true class label is unknown
    """
    # TARGET_DOMAIN_IDX is 1
    source_len = len(class_logits_on_src)
    target_len = len(class_logits_on_trg)
    true_labels_on_src = torch.as_tensor(true_labels_on_src).long()
    if dann_config.IS_UNSUPERVISED:
        true_labels_on_trg = unk_value * torch.ones(target_len, dtype=torch.long, device=device)
    else:
        true_labels_on_trg = torch.as_tensor(true_labels_on_trg).long()
    is_target_on_src = torch.zeros(source_len, dtype=torch.float, device=device)
    is_target_on_trg = torch.ones(target_len, dtype=torch.float, device=device)

    crossentropy = torch.nn.CrossEntropyLoss(ignore_index=unk_value, reduction='sum')
    prediction_loss_on_src = crossentropy(class_logits_on_src, true_labels_on_src)
    prediction_loss_on_trg = crossentropy(class_logits_on_trg, true_labels_on_trg)
    n_known = (true_labels_on_src != unk_value).sum() + \
              (true_labels_on_trg != unk_value).sum()
    prediction_loss = (prediction_loss_on_src + prediction_loss_on_trg) / n_known

    binary_crossentropy = torch.nn.BCEWithLogitsLoss(reduction='sum')
    domain_loss_on_src = binary_crossentropy(logprobs_target_on_src, is_target_on_src)
    domain_loss_on_trg = binary_crossentropy(logprobs_target_on_trg, is_target_on_trg)
    domain_loss = (domain_loss_on_src + domain_loss_on_trg) / (source_len + target_len)
    loss = domain_loss_weight * domain_loss \
           + prediction_loss_weight * prediction_loss
    
    return loss, {
        "domain_loss_on_src": domain_loss_on_src.data.cpu().item() / source_len,
        "domain_loss_on_trg": domain_loss_on_trg.data.cpu().item() / target_len,
        "domain_loss": domain_loss.data.cpu().item(),
        "prediction_loss_on_src": prediction_loss_on_src.data.cpu().item(),
        "prediction_loss_on_trg": prediction_loss_on_trg.data.cpu().item(),
        "prediction_loss": prediction_loss.data.cpu().item()
    }


def calc_rev_grad_alpha(current_iteration,
                        total_iterations,
                        gamma=dann_config.LOSS_GAMMA):
    progress = current_iteration / total_iterations
    lambda_p = 2 / (1 + np.exp(-gamma * progress)) - 1
    return lambda_p


def calc_domain_loss_weight(current_iteration, total_iterations):
    return dann_config.DOMAIN_LOSS


def calc_prediction_loss_weight(current_iteration, total_iterations):
    return dann_config.CLASSIFICATION_LOSS


def loss_DANN(model,
              batch,
              epoch,
              n_epochs,
              device=torch.device('cpu')):
    """
    :param model: model.forward(images) should return dict with keys
        'class' : Tensor, shape = (batch_size, n_classes)  logits  of classes (raw, not logsoftmax)
        'domain': Tensor, shape = (batch_size,) logprob for domain
    :param batch: dict with keys
        'src_images':
        'trg_images':
        'src_classes':np.Array, shape = (batch_size,)
        'trg_classes':np.Array, shape = (batch_size,)
    if true_class is unknown, then class should be dann_config.UNK_VALUE
    :param epoch: current number of iteration
    :param n_epochs: total number of iterations
    :return:
        loss: torch.Tensor,
        losses dict:{
            "domain_loss_on_src"
            "domain_loss_on_trg"
            "domain_loss"
            "prediction_loss_on_src"
            "prediction_loss_on_trg"
            "prediction_loss"
        }
    """
    rev_grad_alpha = calc_rev_grad_alpha(epoch, n_epochs)
    
    model_output = model.forward(batch['src_images'].to(device), rev_grad_alpha)
    class_logits_on_src = model_output['class']
    logprobs_target_on_src = torch.squeeze(model_output['domain'], dim=-1) # TODO: maybe put torch.squeeze in model?

    model_output = model.forward(batch['trg_images'].to(device), rev_grad_alpha)
    class_logits_on_trg = model_output['class']
    logprobs_target_on_trg = torch.squeeze(model_output['domain'], dim=-1)

    domain_loss_weight = calc_domain_loss_weight(epoch, n_epochs)
    prediction_loss_weight = calc_prediction_loss_weight(epoch, n_epochs)
    return _loss_DANN_splitted(
        class_logits_on_src,
        class_logits_on_trg,
        logprobs_target_on_src,
        logprobs_target_on_trg,
        true_labels_on_src=batch['src_classes'],
        true_labels_on_trg=batch['trg_classes'],
        domain_loss_weight=domain_loss_weight,
        prediction_loss_weight=prediction_loss_weight,
        device=device)
