import torch
import utils


def compute_losses(loss_fncs, batch_in, batch_out):
    loss_dict = {}
    for name, fnc in loss_fncs.items():
        if fnc.weight <= 0:
            continue
        loss = fnc(batch_in, batch_out)
        if loss is None:
            continue
        loss_dict[name] = loss
    return loss_dict


def get_loss_grad(batch_in, batch_out, loss_fncs, var_name, loss_name=None):
    """
    get the gradient of selected losses wrt to selected variables
    Which losses and which variable are specified with a list of tuples, grad_pairs
    """
    ## NOTE: need to re-render to re-populate computational graph
    ## in future maybe can also retain graph
    var = batch_out[var_name]
    *dims, C, H, W = var.shape

    var.retain_grad()
    sel_fncs = {loss_name: loss_fncs[loss_name]} if loss_name is not None else loss_fncs
    loss_dict = compute_losses(sel_fncs, batch_in, batch_out)
    if len(loss_dict) < 1:
        return torch.zeros(*dims, 3, H, W, device=var.device), 0

    try:
        sum(loss_dict.values()).backward()
    except:
        pass

    if var.grad is None:
        print("requested grad for {} wrt {} not available".format(loss_name, var_name))
        return torch.zeros(*dims, 3, H, W, device=var.device), 0

    return utils.get_sign_image(var.grad.detach())