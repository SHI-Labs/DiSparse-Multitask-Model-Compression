################################################################################################
# Some Utility Function used for Dynamic Sparsity
################################################################################################

from copy import deepcopy
import torch.nn as nn
import torch
import types
from torch.autograd import Variable
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from scene_net import *
from dataclasses import dataclass
import torch.optim as optim

######################################################################################################
@dataclass
class Decay(object):
    """
    Template decay class
    """
    def __init__(self):
        self.mode = "current"

    def step(self):
        raise NotImplementedError

    def get_dr(self):
        raise NotImplementedError

######################################################################################################
class CosineDecay(Decay):
    """
    Decays a pruning rate according to a cosine schedule.
    Just a wrapper around PyTorch's CosineAnnealingLR.

    :param prune_rate: \alpha described in RigL's paper, initial prune rate (default 0.3)
    :type prune_rate: float
    :param T_max: Max mask-update steps (default 1000)
    :type T_max: int
    :param eta_min: final prune rate (default 0.0)
    :type eta_min: float
    :param last_epoch: epoch to reset annealing. If -1, doesn't reset (default -1).
    :type last_epoch: int
    """
    def __init__(
        self,
        prune_rate: float = 0.5,
        T_max: int = 1000,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        super().__init__()
        self._step = 0
        self.T_max = T_max

        self.sgd = optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max, eta_min, last_epoch
        )

    def step(self, step: int = -1):
        if step >= 0:
            if self._step < self.T_max:
                self.cosine_stepper.step(step)
                self._step = step + 1
            else:
                self._step = self.T_max
            return
        if self._step < self.T_max:
            self.cosine_stepper.step()
            self._step += 1

    def get_dr(self):
        return self.sgd.param_groups[0]["lr"]

######################################################################################################
def googleAI_ERK(net, density, erk_power_scale: float = 0.9):
    """Given the method, returns the sparsity of individual layers as a dict.
    It ensures that the non-custom layers have a total parameter count as the one
    with uniform sparsities. In other words for the layers which are not in the
    custom_sparsity_map the following equation should be satisfied.
    # eps * (p_1 * N_1 + p_2 * N_2) = (1 - default_sparsity) * (N_1 + N_2)
    Args:
      module: 
      density: float, between 0 and 1.
      erk_power_scale: float, if given used to take power of the ratio. Use
        scale<1 to make the erdos_renyi softer.
    Returns:
      density_dict, dict of where keys() are equal to all_masks and individiual
        masks are mapped to the their densities.
    """
    # Obtain masks
    masks = {}
    total_params = 0
    
    for e, (name, layer) in enumerate(net.named_modules()):
        if e == 0:
             continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#             print(name)
            weight = layer.weight
            device = weight.device
            masks[name] = torch.zeros_like(
                weight, dtype=torch.float32, requires_grad=False
            ).to(device)
            total_params += weight.numel()

    # We have to enforce custom sparsities and then find the correct scaling
    # factor.

    is_epsilon_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    dense_layers = set()
    while not is_epsilon_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, mask in masks.items():
            n_param = np.prod(mask.shape)
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density

            if name in dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros

            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones
                # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                raw_probabilities[name] = (
                    np.sum(mask.shape) / np.prod(mask.shape)
                ) ** erk_power_scale
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
#                     print(f"Sparsity of var:{mask_name} had to be set to 0.")
                    dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    density_dict = {}
    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, mask in masks.items():
        n_param = np.prod(mask.shape)
        if name in dense_layers:
            density_dict[name] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[name]
            density_dict[name] = probability_one
#         print(
#             f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
#         )
        total_nonzero += density_dict[name] * mask.numel()
    print(f"Overall sparsity {total_nonzero/total_params}")
    return density_dict

######################################################################################################
def init_prune_net(net):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(hook_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(hook_forward_linear, layer)
    return net

######################################################################################################
def deepcopy_pruned_net(net, copy_net):
    copy_net = get_pruned_init(copy_net)
    copy_net.load_state_dict(net.state_dict())
    return copy_net

######################################################################################################
def erk_init(net, density_dict):
    for name, layer in net.named_modules():
        if name == "backbone.conv1":
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            ratio = density_dict[name]
            prune_ratio = (1 - ratio)
            prune.random_unstructured(layer, name="weight", amount=prune_ratio)
    return net

######################################################################################################
def dynamic_disparse_prune(net, prune_ratio, density_dict, S, device, iteration=None):
    if isinstance(net, nn.DataParallel):
        sparsity_dict = get_sparsity_dict(net.module)
        for name, layer in net.module.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                update_ratio = (density_dict[name] / (1 - sparsity_dict[name])) * (1 - prune_ratio)
                update_ratio = 1 - update_ratio
                update_ratio = min(update_ratio, 1)
                update_ratio = max(update_ratio, 0)
                prune.l1_unstructured(layer, name="weight", amount=update_ratio)
    else:
        sparsity_dict = get_sparsity_dict(net)
        for name, layer in net.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                update_ratio = (density_dict[name] / (1 - sparsity_dict[name])) * (1 - prune_ratio)
                update_ratio = 1 - update_ratio
                update_ratio = min(update_ratio, 1)
                update_ratio = max(update_ratio, 0)
                prune.l1_unstructured(layer, name="weight", amount=update_ratio)
    return net

######################################################################################################
def print_sparsity(prune_net, printing=True):
    # Prine the sparsity
    num = 0
    denom = 0
    ct = 0
    for module in prune_net.modules():
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            if hasattr(module, 'weight'):
                num += torch.sum(module.weight == 0)
                denom += module.weight.nelement()
                if printing:
                    print(
                    f"Layer {ct}", "Sparsity in weight: {:.2f}%".format(
                        100. * torch.sum(module.weight == 0) / module.weight.nelement())
                    )
                ct += 1
    if printing:
        print(f"Model Sparsity Now: {num / denom * 100}")
    return (num / denom).item()

######################################################################################################
def get_pruned_init(net):
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module = prune.identity(module, 'weight')
    return net

######################################################################################################
def deepcopy_pruned_net_v2(net, copy_net):
    weight_masks = {}
    for name, module in net.named_modules():
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            weight_masks[name] = module.weight_mask.data
            module.weight_mask = torch.ones(module.weight_mask.shape).cuda()
    net = pseudo_forward(net)
    for name, module in net.named_modules():
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            prune.remove(module, 'weight')
    copy_net.load_state_dict(net.state_dict())
    copy_net = get_pruned_init(copy_net)
    for name, module in copy_net.named_modules():
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight_mask = weight_masks[name]
    
    copy_net = pseudo_forward(copy_net)
    return copy_net

######################################################################################################
def get_sparsity_dict(net):
    sparsity_dict = {}
    for name, module in net.named_modules():
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            if hasattr(module, 'weight'):
                sparsity_dict[name] = torch.sum(module.weight == 0) / module.weight.nelement()
                sparsity_dict[name] = sparsity_dict[name].item()
    return sparsity_dict

######################################################################################################
def pseudo_forward(net):
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig * module.weight_mask
    return net

def hook_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def hook_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


######################################################################################################
def dynamic_disparse(net, dataset, criterion, gt_batch, density_dict, device, tasks_num_class, tasks):
    parallel_flag = False
    split_num = 2
    if dataset == "taskonomy":
        net = net.module
        parallel_flag = True
        split_num = 8

    test_net = SceneNet(tasks_num_class).to(device)
    test_net = deepcopy_pruned_net(net, test_net)
    
    # Register Hook
    for layer in test_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = torch.ones_like(layer.weight)

    # Get the mask gradients
    grads_abs = {}
    for task in tasks:
        grads_abs[task] = []
    
    sparsity_dict = get_sparsity_dict(net)
    test_net.train()
    # print(split_num)
    for b_idx in range(split_num):
        start = b_idx*(16 // split_num)
        end = (b_idx+1)*(16 // split_num)
        batch_data = {key:val[start:end, :, :, :] for key, val in gt_batch.items() if key != 'name'}
        batch_data['name'] = gt_batch['name']
        
        for i, task in enumerate(tasks):
            torch.cuda.empty_cache()
            test_net.zero_grad()
            preds = test_net.forward(batch_data['img'])
            loss = criterion(preds, batch_data, cur_task=task)
            loss.backward()
            ct = 0

            for name, layer in test_net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if 'backbone' in name or f'task{i+1}' in name:
                        if len(grads_abs[task]) > ct:
                            grads_abs[task][ct] += torch.abs(layer.weight_orig.grad.data)
                        else:
                            grads_abs[task].append(torch.abs(layer.weight_orig.grad.data))
                        ct += 1
                    
    # net = net.cuda()
    net_modules = [m for m in net.modules()]
    
    preds = None
    loss = None
    # Calculate Threshold
    keep_masks = {}
    for task in tasks:
        keep_masks[task] = []
    
    test_net.zero_grad()
    for idx, task in enumerate(tasks):
        ct = 0
        for i, (name, layer) in enumerate(test_net.named_modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if 'backbone' in name or f'task{idx+1}' in name:
                    mask = net_modules[i].weight_mask.to(torch.bool).data
                    grow_ratio = sparsity_dict[name] - (1 - density_dict[name])
                    grow_ratio = max(0, grow_ratio)
                    grow_ratio = min(grow_ratio, 1)
                    if grow_ratio == 0:
                        grow_mask = torch.zeros(mask.shape).bool().cuda()
                        keep_masks[task].append(grow_mask)
                        ct += 1
                        continue
                    elif grow_ratio == 1:
                        grow_mask = torch.ones(mask.shape).bool().cuda()
                        keep_masks[task].append(grow_mask)
                        ct += 1
                        continue
                    try:
                        num_params_to_keep = int(grow_ratio * layer.weight.nelement())
    #                     print(num_params_to_keep)
                        stat = grads_abs[task][ct] * (~mask)
                        flat_stat = torch.flatten(stat)
                        threshold, _ = torch.topk(flat_stat, num_params_to_keep, sorted=True)
                        acceptable_score = threshold[-1]
    #                     print(acceptable_score.item())
                        grow_mask = (stat >= acceptable_score).bool()
                        keep_masks[task].append(grow_mask)
                    except:
                        print(f"Unexpected behavior. Length is {len(flat_stat)}.")
                        grow_mask = torch.ones(mask.shape).bool().cuda()
                        keep_masks[task].append(grow_mask)
                    ct += 1
            
    idxs = [0] * len(tasks)
    ct = 0
    test_net.zero_grad()
    # Copy the masks
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Let's get the union of growing
            if 'backbone' in name:
                final_mask = None
                for i, task in enumerate(tasks):
                    if final_mask is None:
                        final_mask = keep_masks[task][ct].data
                    else:
                        final_mask = final_mask | keep_masks[task][ct].data
                layer.weight_mask = layer.weight_mask.to(torch.bool) | final_mask.to(torch.bool)
                layer.weight_mask = layer.weight_mask.to(torch.float)
                ct += 1
                idxs = [x+1 for x in idxs]

            elif 'task1' in name:
                task_name = tasks[0]
                idx = idxs[0]
                layer.weight_mask = layer.weight_mask.to(torch.bool) | keep_masks[task_name][idx].to(torch.bool)
                layer.weight_mask = layer.weight_mask.to(torch.float)
                ct += 1
                idxs[0] += 1

            elif 'task2' in name:
                task_name = tasks[1]
                idx = idxs[1]
                layer.weight_mask = layer.weight_mask.to(torch.bool) | keep_masks[task_name][idx].to(torch.bool)
                layer.weight_mask = layer.weight_mask.to(torch.float)
                ct += 1
                idxs[1] += 1

            elif 'task3' in name:
                    task_name = tasks[2]
                    idx = idxs[2]
                    layer.weight_mask = layer.weight_mask.to(torch.bool) | keep_masks[task_name][idx].to(torch.bool)
                    layer.weight_mask = layer.weight_mask.to(torch.float)
                    ct += 1
                    idxs[2] += 1

            elif 'task4' in name:
                task_name = tasks[3]
                idx = idxs[3]
                layer.weight_mask = layer.weight_mask.to(torch.bool) | keep_masks[task_name][idx].to(torch.bool)
                layer.weight_mask = layer.weight_mask.to(torch.float)
                ct += 1
                idxs[3] += 1

            elif 'task5' in name:
                task_name = tasks[4]
                idx = idxs[4]
                layer.weight_mask = layer.weight_mask.to(torch.bool) | keep_masks[task_name][idx].to(torch.bool)
                layer.weight_mask = layer.weight_mask.to(torch.float)
                ct += 1
                idxs[4] += 1
            else:
                print(f"Unrecognized Name: {name}!")
    
    net = pseudo_forward(net)
    
    copy_net = SceneNet(tasks_num_class).to(device)
    copy_net = deepcopy_pruned_net_v2(net, copy_net)
    batch_data = None
    net = None
    if parallel_flag:
        return torch.nn.DataParallel(copy_net, device_ids=[0, 1])
    else:
        return copy_net

######################################################################################################