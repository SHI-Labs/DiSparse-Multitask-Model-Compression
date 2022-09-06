################################################################################################
# Contain the compression code of DiSparse and other utility functions
################################################################################################

from copy import deepcopy
import torch.nn as nn
import torch
import types
from torch.autograd import Variable
import torch.nn.utils.prune as prune
import torch.nn.functional as F

################################################################################################
# Overwrite PyTorch forward function for Conv2D and Linear to take the mask into account
def hook_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def hook_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)
################################################################################################
# DiSparse prune in the static setup
# net: model to sparsify, should be of SceneNet class
# criterion: loss function to calculate per task gradients, should be of DiSparse_SceneNetLoss class
# train_loader: dataloader to fetch data batches used to estimate importance
# keep_ratio: how many parameters to keep
# tasks: set of tasks
def disparse_prune_static(net, criterion, train_loader, num_batches, keep_ratio, device, tasks):
    test_net = deepcopy(net)
    grads_abs = {}
    for task in tasks:
        grads_abs[task] = []
    # Register Hook
    for layer in test_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(hook_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(hook_forward_linear, layer)

    # Estimate importance per task in a data-driven manner
    train_iter = iter(train_loader)
    for i in range(num_batches):

        gt_batch = None
        preds = None
        loss = None
        torch.cuda.empty_cache()

        gt_batch = next(train_iter)
        gt_batch["img"] = Variable(gt_batch["img"]).cuda()
        if "seg" in gt_batch:
            gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
        if "depth" in gt_batch:
            gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
        if "normal" in gt_batch:
            gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
        if "keypoint" in gt_batch:
            gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
        if "edge" in gt_batch:
            gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()
        
        for i, task in enumerate(tasks):
            preds = None
            torch.cuda.empty_cache()
            test_net.zero_grad()
            preds = test_net.forward(gt_batch['img'])
            loss = criterion(preds, gt_batch, cur_task=task)
            loss.backward()
            ct = 0
            
            for name, layer in test_net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if 'backbone' in name or f'task{i+1}' in name:
                        if len(grads_abs[task]) > ct:
                            grads_abs[task][ct] += torch.abs(layer.weight_mask.grad.data)
                        else:
                            grads_abs[task].append(torch.abs(layer.weight_mask.grad.data))
                        ct += 1

    preds = None
    loss = None
    # Calculate Threshold
    keep_masks = {}
    for task in tasks:
        keep_masks[task] = []

    # Get importance scores for each task independently
    for i, task in enumerate(tasks):
        cur_grads_abs = grads_abs[task]
        all_scores = torch.cat([torch.flatten(x) for x in cur_grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for g in cur_grads_abs:
            keep_masks[task].append(((g / norm_factor) >= acceptable_score).int())

        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks[task]])))
    
    # Use PyTorch Prune to set hooks
    parameters_to_prune = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parameters_to_prune.append((layer, 'weight'))

    # Use a prune ratio of 0 to set dummy pruning hooks
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0,
    )
    # Compute the final mask
    idxs = [0] * len(tasks)
    ct = 0
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Get the intersection.
            # The following is equivalent to elementwise OR by demorgan
            # Only all tasks agree to prune, we prune
            if 'backbone' in name:
                final_mask = None
                for i, task in enumerate(tasks):
                    if final_mask is None:
                        final_mask = ~keep_masks[task][ct].data
                    else:
                        final_mask = final_mask & (~keep_masks[task][ct].data)
                layer.weight_mask.data = ~final_mask
                ct += 1
                idxs = [x+1 for x in idxs]
                
            elif 'task1' in name:
                task_name = tasks[0]
                idx = idxs[0]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[0] += 1
                
            elif 'task2' in name:
                task_name = tasks[1]
                idx = idxs[1]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[1] += 1
                
            elif 'task3' in name:
                task_name = tasks[2]
                idx = idxs[2]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[2] += 1
                
            elif 'task4' in name:
                task_name = tasks[3]
                idx = idxs[3]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[3] += 1
                
            elif 'task5' in name:
                task_name = tasks[4]
                idx = idxs[4]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[4] += 1

            else:
                print(f"Unrecognized Name: {name}!")
    # Forward
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig * module.weight_mask
            
    print_sparsity(net)
    return net

################################################################################################
# DiSparse prune in the pretrained setup
# net: model to sparsify, should be a SceneNet class
# criterion: loss function to calculate per task gradients, should be DiSparse_SceneNetLoss class
# train_loader: dataloader to fetch data batches used to estimate importance
# keep_ratio: how many parameters to keep
# tasks: set of tasks
def disparse_prune_pretrained(net, criterion, train_loader, num_batches, keep_ratio, device, tasks):
    test_net = deepcopy(net)
    grads_abs = {}
    for task in tasks:
        grads_abs[task] = []

    # Estimate importance per task in a data-driven manner
    train_iter = iter(train_loader)
    for b_idx in range(num_batches):
        gt_batch = None
        preds = None
        loss = None
        torch.cuda.empty_cache()

        gt_batch = next(train_iter)
        gt_batch["img"] = Variable(gt_batch["img"]).cuda()
        if "seg" in gt_batch:
            gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
        if "depth" in gt_batch:
            gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
        if "normal" in gt_batch:
            gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
            
        
        for i, task in enumerate(tasks):
            preds = None
            torch.cuda.empty_cache()
            test_net.zero_grad()
            preds = test_net.forward(gt_batch['img'])
            loss = criterion(preds, gt_batch, cur_task=task)
            loss.backward()
            ct = 0
            
            for name, layer in test_net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if 'backbone' in name or f'task{i+1}' in name:
                        if len(grads_abs[task]) > ct:
                            grads_abs[task][ct] += torch.abs(layer.weight.grad.data)
                        else:
                            grads_abs[task].append(torch.abs(layer.weight.grad.data))
                        
                        if b_idx == (num_batches - 1):
                            grads_abs[task][ct] *= (layer.weight.data).pow(2)
                            
                        ct += 1
                              
    preds = None
    loss = None
    # Calculate Threshold
    keep_masks = {}
    for task in tasks:
        keep_masks[task] = []

    # Get importance scores for each task independently
    for i, task in enumerate(tasks):
        cur_grads_abs = grads_abs[task]
        all_scores = torch.cat([torch.flatten(x) for x in cur_grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for g in cur_grads_abs:
            keep_masks[task].append(((g / norm_factor) >= acceptable_score).int())

        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks[task]])))
    

    # Use PyTorch Prune to set hooks
    parameters_to_prune = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parameters_to_prune.append((layer, 'weight'))

    # Use a prune ratio of 0 to set dummy pruning hooks
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0,
    )

    # Compute the final mask
    idxs = [0] * len(tasks)
    ct = 0
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Get the intersection
            # Only all tasks agree to prune, we prune
            if 'backbone' in name:
                final_mask = None
                for i, task in enumerate(tasks):
                    if final_mask is None:
                        final_mask = ~keep_masks[task][ct].data
                    else:
                        final_mask = final_mask & (~keep_masks[task][ct].data)
                layer.weight_mask.data = ~final_mask
                ct += 1
                idxs = [x+1 for x in idxs]
                
            elif 'task1' in name:
                task_name = tasks[0]
                idx = idxs[0]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[0] += 1
                
            elif 'task2' in name:
                task_name = tasks[1]
                idx = idxs[1]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[1] += 1
            elif 'task3' in name:
                task_name = tasks[2]
                idx = idxs[2]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[2] += 1
                
            elif 'task4' in name:
                task_name = tasks[3]
                idx = idxs[3]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[3] += 1
                
            elif 'task5' in name:
                task_name = tasks[4]
                idx = idxs[4]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[4] += 1
            else:
                print(f"Unrecognized Name: {name}!")
    # Forward
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig * module.weight_mask
            
    print_sparsity(net)
    return net

################################################################################################
# DiSparse prune in the pretrained setup using the l1 score
# net: model to sparsify, should be a SceneNet class
# criterion: loss function to calculate per task gradients, should be DiSparse_SceneNetLoss class
# train_loader: dataloader to fetch data batches used to estimate importance
# keep_ratio: how many parameters to keep
# tasks: set of tasks
def disparse_prune_pretrained_l1(net, criterion, train_loader, num_batches, keep_ratio, device, tasks):
    test_net = deepcopy(net)
    mag_abs = {}
    for task in tasks:
        mag_abs[task] = []
        
    # Estimate importance per task in a data-driven manner
    train_iter = iter(train_loader)
    idxs = [0] * len(tasks)
    ct = 0
    
    for i, task in enumerate(tasks):
        for name, layer in test_net.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if 'backbone' in name or f'task{i+1}' in name:
                    if len(mag_abs[task]) > ct:
                        mag_abs[task][ct] += torch.abs(layer.weight.data)
                    else:
                        mag_abs[task].append(torch.abs(layer.weight.data))
                    ct += 1
    
    # Calculate Threshold
    keep_masks = {}
    for task in tasks:
        keep_masks[task] = []

    # Get importance scores for each task independently
    for i, task in enumerate(tasks):
        cur_mag_abs = mag_abs[task]
        all_scores = torch.cat([torch.flatten(x) for x in cur_mag_abs])
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        for g in cur_mag_abs:
            keep_masks[task].append(((g) >= acceptable_score).int())
            
        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks[task]])))
    
    
    parameters_to_prune = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parameters_to_prune.append((layer, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0,
    )

    idxs = [0] * len(tasks)
    ct = 0
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Get the intersection
            # Only all tasks agree to prune, we prune
            if 'backbone' in name:
                final_mask = None
                for i, task in enumerate(tasks):
                    if final_mask is None:
                        final_mask = ~keep_masks[task][ct].data
                    else:
                        final_mask = final_mask & (~keep_masks[task][ct].data)
                layer.weight_mask.data = ~final_mask
                ct += 1
                idxs = [x+1 for x in idxs]
                
            elif 'task1' in name:
                task_name = tasks[0]
                idx = idxs[0]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[0] += 1
                
            elif 'task2' in name:
                task_name = tasks[1]
                idx = idxs[1]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[1] += 1
                
            elif 'task3' in name:
                task_name = tasks[2]
                idx = idxs[2]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[2] += 1
                
            elif 'task4' in name:
                task_name = tasks[3]
                idx = idxs[3]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[3] += 1
                
            elif 'task5' in name:
                task_name = tasks[4]
                idx = idxs[4]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[4] += 1
                
            else:
                print(f"Unrecognized Name: {name}!")
                
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig * module.weight_mask
    print_sparsity(net)
    
    return net

################################################################################################
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
    return num / denom

################################################################################################
def get_pruned_init(net):
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module = prune.identity(module, 'weight')
    return net

################################################################################################
def deepcopy_pruned_net(net, copy_net):
    copy_net = get_pruned_init(copy_net)
    copy_net.load_state_dict(net.state_dict())
    return copy_net

################################################################################################
def get_sparsity_dict(net):
    sparsity_dict = {}
    for name, module in net.named_modules():
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            if hasattr(module, 'weight'):
                sparsity_dict[name] = torch.sum(module.weight == 0) / module.weight.nelement()
                sparsity_dict[name] = sparsity_dict[name].item()
    return sparsity_dict

################################################################################################
def pseudo_forward(net):
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig * module.weight_mask
    return net

################################################################################################