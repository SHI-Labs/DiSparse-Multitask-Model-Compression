################################################################################################
# Evaluation Code
################################################################################################

from sklearn.metrics import confusion_matrix
import numpy as np
from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os

################################################################################################
class SceneNetEval():
    def __init__(self, device, tasks, tasks_num_class, image_shape, dataset, data_root=None):
        self.device = device
        self.tasks = tasks
        self.tasks_num_class = tasks_num_class
        self.image_shape = image_shape
        self.dataset = dataset

        if 'seg' in self.tasks:
            self.seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
        self.reset_records()
        if dataset == "taskonomy":
            assert data_root is not None
            weight = torch.from_numpy(np.load(os.path.join(data_root, 'semseg_prior_factor.npy'))).to(self.device).float()
            self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        elif dataset == "nyuv2":
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        elif dataset == "cityscapes":
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            print("Unrecocgnized Dataset.")
            exit()
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss()
    ################################################################################################
    def reset_records(self):
        self.records = {}
        if 'seg' in self.tasks:
            self.records['seg'] = {'mIoUs': [], 'pixelAccs': [], 'errs': [], 'conf_mat': np.zeros((self.seg_num_class, self.seg_num_class)),
                              'labels': np.arange(self.seg_num_class), 'gts': [], 'preds': []}
        if 'sn' in self.tasks:
            self.records['sn'] = {'cos_similaritys': []}
        if 'depth' in self.tasks:
            self.records['depth'] = {'abs_errs': [], 'rel_errs': [], 'sq_rel_errs': [], 'ratios': [], 'rms': [], 'rms_log': []}
        if 'keypoint' in self.tasks:
            self.records['keypoint'] = {'errs': []}
        if 'edge' in self.tasks:
            self.records['edge'] = {'errs': []}
    ################################################################################################
    def resize_results(self):
        new_shape = self.image_shape
        if 'seg' in self.tasks:
            self.seg_output = F.interpolate(self.seg_pred, size=new_shape)
        if 'sn' in self.tasks:
            self.sn_output = F.interpolate(self.sn_pred, size=new_shape)
        if 'depth' in self.tasks:
            self.depth_output = F.interpolate(self.depth_pred, size=new_shape)
            if self.dataset == "taskonomy":
                self.depth_mask = F.interpolate(self.depth_mask.float(), size=new_shape)
        if 'keypoint' in self.tasks:
            self.keypoint_output = F.interpolate(self.keypoint_pred, size=new_shape)
        if 'edge' in self.tasks:
            self.edge_output = F.interpolate(self.edge_pred, size=new_shape)
    ################################################################################################
    def parse_inputs(self, preds, targets):
        if 'seg' in self.tasks:
            self.seg_pred = preds[self.tasks.index('seg')]
            self.seg = targets['seg']
        if 'sn' in self.tasks:
            self.sn_pred = preds[self.tasks.index('sn')]
            self.normal = targets['normal']
        if 'depth' in self.tasks:
            self.depth_pred = preds[self.tasks.index('depth')]
            self.depth = targets['depth']
            if self.dataset == "taskonomy":
                self.depth_mask = targets['depth_mask']
        if 'keypoint' in self.tasks:
            self.keypoint_pred = preds[self.tasks.index('keypoint')]
            self.keypoint = targets['keypoint']
        if 'edge' in self.tasks:
            self.edge_pred = preds[self.tasks.index('edge')]
            self.edge = targets['edge']
    ####################################################################################################
    def seg_error(self):
        gt = self.seg.view(-1)
        labels = gt < self.seg_num_class
        gt = gt[labels].int()

        logits = self.seg_output.permute(0, 2, 3, 1).contiguous().view(-1, self.seg_num_class)
        logits = logits[labels]
        err = self.cross_entropy(logits, gt.long())

        prediction = torch.argmax(self.seg_output, dim=1)
        prediction = prediction.unsqueeze(1)

        # pixel acc
        prediction = prediction.view(-1)
        prediction = prediction[labels].int()
        pixelAcc = (gt == prediction).float().mean()

        return prediction.cpu().numpy(), gt.cpu().numpy(), pixelAcc, err.cpu().numpy()
    ################################################################################################
    def normal_error(self):
        # normalized, ignored gt and prediction
        prediction = self.sn_output.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        gt = self.normal.permute(0, 2, 3, 1).contiguous().view(-1, 3)

        labels = gt.max(dim=1)[0] != 255
        if hasattr(self, 'normal_mask'):
            gt_mask = self.normal_mask.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            labels = labels and gt_mask.int() == 1

        gt = gt[labels]
        prediction = prediction[labels]

        gt = F.normalize(gt.float(), dim=1)
        prediction = F.normalize(prediction, dim=1)

        cos_similarity = self.cosine_similiarity(gt, prediction)

        return cos_similarity.cpu().numpy()
    ################################################################################################
    def depth_error(self):
        if self.dataset == "cityscapes":
            binary_mask = (torch.sum(self.depth, dim=1) > 3 * 1e-5).unsqueeze(1).to(self.device)
        else:
            binary_mask = (self.depth != 255) * (self.depth_mask.int() == 1)
        
        depth_output_true = self.depth_output.masked_select(binary_mask)
        depth_gt_true = self.depth.masked_select(binary_mask)
        abs_err = torch.abs(depth_output_true - depth_gt_true)
        rel_err = torch.abs(depth_output_true - depth_gt_true) / depth_gt_true
        sq_rel_err = torch.pow(depth_output_true - depth_gt_true, 2) / depth_gt_true
        abs_err = torch.sum(abs_err) / torch.nonzero(binary_mask).size(0)
        rel_err = torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)
        sq_rel_err = torch.sum(sq_rel_err) / torch.nonzero(binary_mask).size(0)
        # calcuate the sigma
        term1 = depth_output_true / depth_gt_true
        term2 = depth_gt_true / depth_output_true
        ratio = torch.max(torch.stack([term1, term2], dim=0), dim=0)
        # calcualte rms
        rms = torch.pow(depth_output_true - depth_gt_true, 2)
        rms_log = torch.pow(torch.log10(depth_output_true + 1e-7) - torch.log10(depth_gt_true + 1e-7), 2)

        return abs_err.cpu().numpy(), rel_err.cpu().numpy(), sq_rel_err.cpu().numpy(), ratio[0].cpu().numpy(), \
               rms.cpu().numpy(), rms_log.cpu().numpy()
    
    ################################################################################################
    def keypoint_error(self):
        binary_mask = (self.keypoint != 255).to(self.device)
        keypoint_output_true = self.keypoint_output.masked_select(binary_mask)
        keypoint_gt_true = self.keypoint.masked_select(binary_mask)
        abs_err = torch.abs(keypoint_output_true - keypoint_gt_true).mean()
        return abs_err.cpu().numpy()

    ################################################################################################
    def edge_error(self):
        binary_mask = (self.edge != 255).to(self.device)
        edge_output_true = self.edge_output.masked_select(binary_mask)
        edge_gt_true = self.edge.masked_select(binary_mask)
        abs_err = torch.abs(edge_output_true - edge_gt_true).mean()
        return abs_err.cpu().numpy()
    ####################################################################################################
    def calculate_error(self, preds, targets):
        metrics = {}
        self.parse_inputs(preds, targets)
        self.resize_results()
        
        if 'seg' in self.tasks:
            metrics['seg'] = {}
            pred, gt, pixelAcc, err = self.seg_error()
            metrics['seg']['pred'] = pred
            metrics['seg']['gt'] = gt
            metrics['seg']['pixelAcc'] = pixelAcc.cpu().numpy()
            metrics['seg']['err'] = err
        if 'sn' in self.tasks:
            metrics['sn'] = {}
            cos_similarity = self.normal_error()
            metrics['sn']['cos_similarity'] = cos_similarity
        if 'depth' in self.tasks:
            metrics['depth'] = {}
            abs_err, rel_err, sq_rel_err, ratio, rms, rms_log = self.depth_error()
            metrics['depth']['abs_err'] = abs_err
            metrics['depth']['rel_err'] = rel_err
            metrics['depth']['sq_rel_err'] = sq_rel_err
            metrics['depth']['ratio'] = ratio
            metrics['depth']['rms'] = rms
            metrics['depth']['rms_log'] = rms_log
        
        if 'keypoint' in self.tasks:
            metrics['keypoint'] = {}
            err = self.keypoint_error()
            metrics['keypoint']['err'] = err
            
        if 'edge' in self.tasks:
            metrics['edge'] = {}
            err = self.edge_error()
            metrics['edge']['err'] = err
            
        return metrics
    ####################################################################################################
    def update_records(self, preds, targets):
        metrics = self.calculate_error(preds, targets)
        if 'seg' in self.tasks:
            self.records['seg']['gts'].append(metrics['seg']['gt'])
            self.records['seg']['preds'].append(metrics['seg']['pred'])
            new_mat = confusion_matrix(y_true = metrics['seg']['gt'], y_pred = metrics['seg']['pred'], labels = self.records['seg']['labels'])

            assert (self.records['seg']['conf_mat'].shape == new_mat.shape)
            self.records['seg']['conf_mat'] += new_mat
            self.records['seg']['pixelAccs'].append(metrics['seg']['pixelAcc'])
            self.records['seg']['errs'].append(metrics['seg']['err'])
        if 'sn' in self.tasks:
            self.records['sn']['cos_similaritys'].append(metrics['sn']['cos_similarity'])
        if 'depth' in self.tasks:
            self.records['depth']['abs_errs'].append(metrics['depth']['abs_err'])
            self.records['depth']['rel_errs'].append(metrics['depth']['rel_err'])
            self.records['depth']['sq_rel_errs'].append(metrics['depth']['sq_rel_err'])
            self.records['depth']['ratios'].append(metrics['depth']['ratio'])
            self.records['depth']['rms'].append(metrics['depth']['rms'])
            self.records['depth']['rms_log'].append(metrics['depth']['rms_log'])
        if 'keypoint' in self.tasks:
            self.records['keypoint']['errs'].append(metrics['keypoint']['err'])
        if 'edge' in self.tasks:
            self.records['edge']['errs'].append(metrics['edge']['err'])

    ################################################################################################
    def get_final_metrics(self, net, test_loader):
        val_metrics = {}
        batch_size = []
        self.reset_records()
        # Get the predictions
        with torch.no_grad():
            net.eval()
            for i, gt_batch in enumerate(test_loader):
                if i % 100 == 0:
                    print(f"{i}th image processed")
#                 gt_batch = test_dataset[i]
                gt_batch["img"] = Variable(gt_batch["img"]).cuda()
                if "seg" in gt_batch:
                    gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
                if "depth" in gt_batch:
                    gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
                    if self.dataset == "taskonomy":
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
                self.update_records(preds, gt_batch)
                batch_size.append(len(gt_batch['img']))

        # Process the records
        if 'seg' in self.tasks:
            val_metrics['seg'] = {}
            jaccard_perclass = []
            for i in range(self.seg_num_class):
                if not self.records['seg']['conf_mat'][i, i] == 0:
                    jaccard_perclass.append(self.records['seg']['conf_mat'][i, i] / (np.sum(self.records['seg']['conf_mat'][i, :]) +
                                                                                np.sum(self.records['seg']['conf_mat'][:, i]) -
                                                                                self.records['seg']['conf_mat'][i, i]))

            val_metrics['seg']['mIoU'] = np.sum(jaccard_perclass) / len(jaccard_perclass)

            val_metrics['seg']['Pixel Acc'] = (np.array(self.records['seg']['pixelAccs']) * np.array(batch_size)).sum() / sum(
                batch_size)

            val_metrics['seg']['err'] = (np.array(self.records['seg']['errs']) * np.array(batch_size)).sum() / sum(batch_size)

        if 'sn' in self.tasks:
            val_metrics['sn'] = {}
            overall_cos = np.clip(np.concatenate(self.records['sn']['cos_similaritys']), -1, 1)

            angles = np.arccos(overall_cos) / np.pi * 180.0
            val_metrics['sn']['Angle Mean'] = np.mean(angles)
            val_metrics['sn']['Angle Median'] = np.median(angles)
            val_metrics['sn']['Angle 11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
            val_metrics['sn']['Angle 22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
            val_metrics['sn']['Angle 30'] = np.mean(np.less_equal(angles, 30.0)) * 100
            val_metrics['sn']['Angle 45'] = np.mean(np.less_equal(angles, 45.0)) * 100

        if 'depth' in self.tasks:
            val_metrics['depth'] = {}
            self.records['depth']['abs_errs'] = np.stack(self.records['depth']['abs_errs'], axis=0)
            self.records['depth']['rel_errs'] = np.stack(self.records['depth']['rel_errs'], axis=0)
            self.records['depth']['ratios'] = np.concatenate(self.records['depth']['ratios'], axis=0)
            val_metrics['depth']['abs_err'] = (self.records['depth']['abs_errs'] * np.array(batch_size)).sum() / sum(batch_size)
            val_metrics['depth']['rel_err'] = (self.records['depth']['rel_errs'] * np.array(batch_size)).sum() / sum(batch_size)
            val_metrics['depth']['sigma_1.25'] = np.mean(np.less_equal(self.records['depth']['ratios'], 1.25)) * 100
            val_metrics['depth']['sigma_1.25^2'] = np.mean(np.less_equal(self.records['depth']['ratios'], 1.25 ** 2)) * 100
            val_metrics['depth']['sigma_1.25^3'] = np.mean(np.less_equal(self.records['depth']['ratios'], 1.25 ** 3)) * 100
        
        if 'keypoint' in self.tasks:
            val_metrics['keypoint'] = {}
            val_metrics['keypoint']['err'] = (np.array(self.records['keypoint']['errs']) * np.array(batch_size)).sum() / sum(
                batch_size)

        if 'edge' in self.tasks:
            val_metrics['edge'] = {}
            val_metrics['edge']['err'] = (np.array(self.records['edge']['errs']) * np.array(batch_size)).sum() / sum(
                batch_size)
        
        return val_metrics
    
