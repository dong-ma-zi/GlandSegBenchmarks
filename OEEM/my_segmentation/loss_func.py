import torch.nn.functional as F

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def cross_entropy(pred,
                  label,
                  criterion,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  custom_str=''):
    """The wrapper function for :func:`F.cross_entropy`"""

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        # reduction='mean',
        ignore_index=ignore_index)
    # loss = criterion(pred, label)

    # online easy example mining
    if 'oeem' in custom_str:

        # normalized loss
        # weight = torch.ones_like(loss)
        metric = -loss.detach().reshape((loss.shape[0], loss.shape[1] * loss.shape[2]))
        weight = F.softmax(metric, 1)
        weight = weight / weight.mean(1).reshape((-1, 1))
        weight = weight.reshape((loss.shape[0], loss.shape[1], loss.shape[2]))

        # apply oeem on images of multiple labels
        for i in range(label.shape[0]):
            tag = set(label[i].reshape(label.shape[1] * label.shape[2]).tolist()) - {255}
            if len(tag) <= 1:
                weight[i] = 1

    # apply weights and reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss