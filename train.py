################################################################################################
# Training Utility Function
################################################################################################

from torch.autograd import Variable
import torch

def train(net, dataset, criterion, optimizer, scheduler, train_loader, test_loader, network_name, batch_update, max_iters=20000, save_model=True, log_file=None, method="foo", dest="/data"):
    iteration = 0
    epoch = 0
    flag = False
    best_test_loss = None
    lr = 0
    save_int = 10000 if dataset == "taskonomy" else 5000
    test_int = 10000 if dataset == "taskonomy" else 50
    
    while True:
        total_loss = 0
        for i, gt_batch in enumerate(train_loader):
            net.train()
            gt_batch["img"] = Variable(gt_batch["img"]).cuda()
            if "seg" in gt_batch:
                gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
            if "depth" in gt_batch:
                gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
                if dataset == "taskonomy":
                    if 'depth_mask' in gt_batch.keys():
                        gt_batch["depth_mask"] = Variable(gt_batch["depth_mask"]).cuda()
                    else:
                        print("No Depth Mask Existing. Please check")
                        gt_batch["depth_mask"] = Variable(torch.ones(gt_batch["depth"].shape)).cuda()
            if "normal" in gt_batch:
                gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
            if "keypoint" in gt_batch:
                gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
            if "edge" in gt_batch:
                gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()
                
            # get the preds
            preds = net(gt_batch["img"])
            loss = criterion(preds, gt_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (iteration+1) % 10 == 0:
                print(f'{method}: Epoch [%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                      % (epoch, iteration+1, max_iters, loss.item(), total_loss / (i+1)))
                log_file.write(f'{method}: Epoch [%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                      % (epoch, iteration+1, max_iters, loss.item(), total_loss / (i+1)))
                log_file.write("\n")
            # Save the model
            if save_model:
                if iteration % save_int == 0:
                    print("Save checkpoint.")
                    torch.save(net.state_dict(), f"{dest}/{iteration}th_{network_name}.pth")
            iteration += 1
            
            scheduler.step()
            if iteration % 100 == 0:
                print(scheduler.get_last_lr())
            if iteration > max_iters:
                flag = True
                break
            
            # Validate on test dataset
            if iteration % test_int == 0:
                net.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    for i, gt_batch in enumerate(test_loader):
                        gt_batch["img"] = Variable(gt_batch["img"]).cuda()
                        if "seg" in gt_batch:
                            gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
                        if "depth" in gt_batch:
                            gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
                            if dataset == "taskonomy":
                                if 'depth_mask' in gt_batch.keys():
                                    gt_batch["depth_mask"] = Variable(gt_batch["depth_mask"]).cuda()
                                else:
                                    print("No Depth Mask Existing. Please check")
                                    gt_batch["depth_mask"] = Variable(torch.ones(gt_batch["depth"].shape)).cuda()
                        if "normal" in gt_batch:
                            gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
                        if "keypoint" in gt_batch:
                            gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
                        if "edge" in gt_batch:
                            gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()

                        preds = net(gt_batch["img"])
                        loss = criterion(preds, gt_batch)
                        test_loss += loss.item()
                    test_loss /= len(test_loader)
                    print(f"{method}: TEST LOSS on {epoch}th epoch:", test_loss)
                    log_file.write(f"{method}: TEST LOSS on {epoch}th epoch: {test_loss}")
                    log_file.write("\n")
                    
                    if save_model:
                        if best_test_loss is None:
                            best_test_loss = test_loss
                        elif test_loss < best_test_loss:
                            torch.save(net.state_dict(), f"{dest}/best_{network_name}.pth")
                            best_test_loss = test_loss
                        
        epoch += 1

        # End Training
        if flag:
            break
    return net


from dynamic_prune import *
def disparse_dynamic_train(net, dataset, criterion, amp_criterion, optimizer, scheduler, train_loader, test_loader, network_name, batch_update, S, config_dict, device = "cuda", max_iters=None, save_model=True, log_file=None, method="rigl", dest="/data"):
    prune_rate, end, interval, init_lr, weight_decay, tasks_num_class, tasks = \
    config_dict["prune_rate"], config_dict["end"], config_dict["interval"], config_dict["init_lr"], config_dict["weight_decay"], config_dict["tasks_num_class"], config_dict["tasks"]
    decay_freq, decay_rate = config_dict["decay_freq"], config_dict["decay_rate"]
    if dataset == "taskonomy":
        density_dict = googleAI_ERK(net.module, S)
        net = erk_init(net.module, density_dict)
        net = nn.DataParallel(net, device_ids=[0, 1])
    else:
        density_dict = googleAI_ERK(net, S)
        net = erk_init(net, density_dict)
    print_sparsity(net)
    decay = CosineDecay(prune_rate = prune_rate, T_max = end)
    
    # if max_iters is None:
    #     max_iters = MAX_ITERS
    iteration = 0
    epoch = 0
    flag = False
    best_test_loss = None
    lr = 0
    save_int = 10000 if dataset == "taskonomy" else 5000
    test_int = 10000 if dataset == "taskonomy" else 50
    
    while True:
        total_loss = 0
        for i, gt_batch in enumerate(train_loader):
            net.train()
            gt_batch["img"] = Variable(gt_batch["img"]).cuda()
            if "seg" in gt_batch:
                gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
            if "depth" in gt_batch:
                gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
                if dataset == "taskonomy":
                    if 'depth_mask' in gt_batch.keys():
                        gt_batch["depth_mask"] = Variable(gt_batch["depth_mask"]).cuda()
                    else:
                        print("No Depth Mask Existing. Please check")
                        gt_batch["depth_mask"] = Variable(torch.ones(gt_batch["depth"].shape)).cuda()
            if "normal" in gt_batch:
                gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
            if "keypoint" in gt_batch:
                gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
            if "edge" in gt_batch:
                gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()
            
            
            # Update connection
            if iteration % interval == 0 and iteration < end:
                print(f"Prune Rate: {decay.get_dr()}")
                s = print_sparsity(net, False)
                print(f"Sparsity Before Prune: {s}")
                net = dynamic_disparse_prune(net, decay.get_dr(), density_dict, 1-S, device)
                s = print_sparsity(net, False)
                print(f"Sparsity After Prune: {s}")
                net = dynamic_disparse(net, dataset, amp_criterion, gt_batch, density_dict, device, tasks_num_class, tasks)
                new_optimizer = torch.optim.Adam(net.parameters(), lr = init_lr, weight_decay = weight_decay)
                new_optimizer.load_state_dict(optimizer.state_dict())
                optimizer = new_optimizer
                new_optimizer = None
                new_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_freq, gamma=decay_rate)
                new_scheduler.load_state_dict(scheduler.state_dict())
                scheduler = new_scheduler
                new_scheduler = None
            
            if iteration == end:
                print("Updating Session End.")
                print(f"Prune Rate: {decay.get_dr()}")
                s = print_sparsity(net, False)
                print(f"Sparsity Before Prune: {s}")
                net = dynamic_disparse_prune(net, decay.get_dr(), density_dict, 1-S, device, iteration)
                net = copy_v2(net)
                s = print_sparsity(net, False)
                print(f"Sparsity After Prune: {s}")
                new_optimizer = torch.optim.Adam(net.parameters(), lr = init_lr, weight_decay = weight_decay)
                new_optimizer.load_state_dict(optimizer.state_dict())
                optimizer = new_optimizer
                new_optimizer = None
                new_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_freq, gamma=decay_rate)
                new_scheduler.load_state_dict(scheduler.state_dict())
                scheduler = new_scheduler
                new_scheduler = None
                net.train()
                s = print_sparsity(net, False)
                print(f"Final Sparsity: {s}")
                
            torch.cuda.empty_cache()    
            # get the preds
            preds = net(gt_batch["img"])
            loss = criterion(preds, gt_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (iteration+1) % 10 == 0:
                print(f'{method}: Epoch [%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                      % (epoch, iteration+1, max_iters, loss.item(), total_loss / (i+1)))
                log_file.write(f'{method}: Epoch [%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                      % (epoch, iteration+1, max_iters, loss.item(), total_loss / (i+1)))
                log_file.write("\n")
            # Save the model
            if save_model:
                if iteration % save_int == 0:
                    print("Save checkpoint.")
                    torch.save(net.state_dict(), f"{dest}/{iteration}th_{network_name}.pth")
            iteration += 1
            
            scheduler.step()
            decay.step()
            
            if iteration % 100 == 0:
                print(scheduler.get_last_lr())
            if iteration > max_iters:
                flag = True
                break
            
            # Test on test dataset
            if iteration % test_int == 0:
                net.eval()
                with torch.no_grad():
                    test_loss = 0.0
                    for i, gt_batch in enumerate(test_loader):
                        gt_batch["img"] = Variable(gt_batch["img"]).cuda()
                        if "seg" in gt_batch:
                            gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
                        if "depth" in gt_batch:
                            gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
                            if dataset == "taskonomy":
                                if 'depth_mask' in gt_batch.keys():
                                    gt_batch["depth_mask"] = Variable(gt_batch["depth_mask"]).cuda()
                                else:
                                    print("No Depth Mask Existing. Please check")
                                    gt_batch["depth_mask"] = Variable(torch.ones(gt_batch["depth"].shape)).cuda()
                        if "normal" in gt_batch:
                            gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
                        if "keypoint" in gt_batch:
                            gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
                        if "edge" in gt_batch:
                            gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()

                        preds = net(gt_batch["img"])
                        loss = criterion(preds, gt_batch)

                        test_loss += loss.item()
                    test_loss /= len(test_loader)
                    print(f"{method}: TEST LOSS on {epoch}th epoch:", test_loss)
                    log_file.write(f"{method}: TEST LOSS on {epoch}th epoch: {test_loss}")
                    log_file.write("\n")
                    
                    if save_model:
                        if best_test_loss is None:
                            best_test_loss = test_loss
                        elif test_loss < best_test_loss:
                            torch.save(net.state_dict(), f"{dest}/best_{network_name}.pth")
                            best_test_loss = test_loss
                        
        epoch += 1

        # End Training
        if flag:
            break
    return net