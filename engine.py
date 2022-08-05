# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import json
from json import JSONEncoder
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as transforms
from PIL import Image,ImageDraw, ImageFont
import util.box_ops as box_ops
import torchvision.transforms.functional as F


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    # samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for samples, attens, support_imgs, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        attens = attens.to(device)
        support_imgs = support_imgs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples,support_imgs,attens.decompose()[0])
        loss_dict = criterion(outputs, targets,attens)
        weight_dict = criterion.weight_dict
        #print(weight_dict)
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
 
        # for n,p in model.named_parameters():
        #     if 'support' in n and 'layers' in n and '0' in n and 'multihead_attn' in n and 'in_proj_weight' in n:
        #         print(p)
        #         print(p.grad)

        # print('--------------------------------------')
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # for n,p in model.named_parameters():
        #     if 'support' in n and 'layers' in n and '0' in n and 'multihead_attn' in n and 'in_proj_weight' in n:
        #         print(p)
                # print(p.grad)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        #samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples,)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

@torch.no_grad()
def evaluate_caltech_map(model, criterion, postprocessors, data_loader, device, output_dir_caltech):
    model.eval()
    criterion.eval()

    # base_ds = get_coco_api_from_dataset(train_datasets)


    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())

    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    predictions,groundtruths = [],[]
    dt_anns = []
    i = 0
    idx = 0

    cocolike_predictions = []
    anns = []
    img_ids = []
    for samples, attens, support_imgs, targets in metric_logger.log_every(data_loader, 100, header):
        # if i == 1000:
        #     break
        # i += 1
        samples = samples.to(device)
        support_imgs = support_imgs.to(device)
        attens = attens.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples,support_imgs,attens.decompose()[0],mode='test')
        if 'pred_atten' in outputs:
            GetAttentionMap(attens,outputs['pred_atten'],idx,output_dir_caltech)
            idx = idx + 1
        loss_dict = criterion(outputs, targets,attens)


        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        loss_scaled_output = ["loss_ce", "loss_bbox", "loss_giou","loss_attens"] #,"loss_attention"
        loss_unscaled_output = ["loss_ce_unscaled", "class_error_unscaled", "loss_bbox_unscaled", "loss_giou_unscaled", "cardinality_error_unscaled"] #, "loss_giou_unscaled", "loss_attention_unscaled"
        
        loss_dict_reduced_scaled_out = {x: loss_dict_reduced_scaled.get(x) for x in loss_scaled_output}
        loss_dict_reduced_unscaled_out = {x: loss_dict_reduced_unscaled.get(x) for x in loss_unscaled_output}
               
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled_out,
                             **loss_dict_reduced_unscaled_out)                              
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)  

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        pred_boxes = []
        gt_boxes = []
        b,c,w,h = samples.decompose()[0].shape
        for target, result in zip(targets,results):
            predictions.append(result)
            groundtruths.append(target)
            # print(result)

            box = box_ops.xyxy_to_xywh(result['boxes']).tolist()  # xywh
            # box = prediction['boxes'].tolist()
            score = result['scores'].tolist()  # (#objs,)
            # label = torch.zeros_like(prediction['labels']).tolist()  # (#objs,)
            label = result['labels'].tolist()
            # for predcls, we set label and score to groundtruth

            image_id = target['image_id'].item()
            image_id_np = np.asarray([image_id]*len(box))

            cocolike_predictions.append(
                np.column_stack((image_id_np, box, score, label))
            )

            labels = target['labels'].tolist()  # integer
            orig_size = target['orig_size']
            boxes = box_ops.box_cxcywh_to_xywh(target['boxes']*torch.tensor([orig_size[1],orig_size[0],orig_size[1],orig_size[0]],device=device)).tolist()  # xywh
            for cls, box in zip(labels, boxes):
                anns.append({
                    'area': box[3] * box[2],
                    'bbox': [box[0], box[1], box[2], box[3]],  # xywh
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': image_id,
                    'iscrowd': 0,
                })
            
            img_ids.append({'id':image_id})

        
            labels = result['labels'].tolist()  # integer
            scores = result['scores'].tolist()
            orig_size = target['orig_size']

            # For visulization
            pred_boxes.append(result['boxes'].tolist())
            gt_b = box_ops.box_cxcywh_to_xyxy(target['boxes']*torch.tensor([h,w,h,w],device=device)).tolist()
            gt_boxes.append(gt_b)

            boxes = box_ops.xyxy_to_xywh(result['boxes']).tolist()  # xywh
            # boxes = dt['boxes'].tolist()
            for cls, box, score in zip(labels, boxes, scores):
                if score > 0.1 and box[3] >= 40:
                    dt_anns.append({
                        'area': box[3] * box[2],
                        'bbox': [box[0], box[1], box[2], box[3]],  # xywh
                        'id': len(dt_anns),
                        'image_id': image_id,
                        'category_id':1,
                        'score': score,
                    })


        plot_bbox_on_img(samples.decompose()[0],pred_boxes,gt_boxes,output_dir_caltech,idx)
        idx = idx + 1
        

    fauxcoco = COCO()
    # cocolike_predictions = []

    # for image_id, prediction in enumerate(predictions):

    #     box = box_ops.xyxy_to_xywh(prediction['boxes']).tolist()  # xywh
    #     # box = prediction['boxes'].tolist()
    #     score = prediction['scores'].tolist()  # (#objs,)
    #     # label = torch.zeros_like(prediction['labels']).tolist()  # (#objs,)
    #     label = prediction['labels'].tolist()
    #     # for predcls, we set label and score to groundtruth

    #     image_id = np.asarray([image_id] * len(box))
    #     cocolike_predictions.append(
    #         np.column_stack((image_id, box, score, label))
    #     )

    # with open("input/data.json",'r') as load_f:
    #     fauxcoco.dataset = json.load(load_f)

    # anns = []

    # for image_id, gt in enumerate(groundtruths):
    #     labels = torch.zeros_like(gt['labels']).tolist()  # integer
    #     orig_size = gt['orig_size']
    #     boxes = box_ops.box_cxcywh_to_xywh(gt['boxes']*torch.tensor([orig_size[1],orig_size[0],orig_size[1],orig_size[0]],device=device)).tolist()  # xywh
    #     for cls, box in zip(labels, boxes):
    #         anns.append({
    #             'area': box[3] * box[2],
    #             'bbox': [box[0], box[1], box[2], box[3]],  # xywh
    #             'category_id': cls,
    #             'id': len(anns),
    #             'image_id': image_id,
    #             'iscrowd': 0,
    #         })

    with open("val_gt.json",'w') as f:
        json.dump(anns,f,cls=NumpyArrayEncoder)

    # fauxcoco.dataset = {
    #     'info': {'description': 'use coco script for vg detection evaluation'},
    #     'images': [{'id': i} for i in range(len(groundtruths))],
    #     'categories': [
    #         {'supercategory': 'person', 'id': i, 'name': i}
    #         for i in range(5)
    #     ],
    #     'annotations': anns,
    # }

    fauxcoco.dataset = {
        'info': {'description': 'use coco script for vg detection evaluation'},
        'images': img_ids,
        'categories': [
            {'supercategory': 'person', 'id': 1, 'name': 1}
        ],
        'annotations': anns,
    }


    fauxcoco.createIndex()
    cocolike_predictions = np.concatenate(cocolike_predictions, 0)
    # evaluate via coco API
    res = fauxcoco.loadRes(cocolike_predictions)
    coco_evaluator = COCOeval(fauxcoco, res, 'bbox')
    # coco_evaluator.params.imgIds = list(range(len(predictions)))
    coco_evaluator.params.imgIds = [int(list(s.values())[0]) for s in img_ids]
    coco_evaluator.evaluate()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.stats.tolist()


    with open("val_dt.json",'w') as f:
        json.dump(dt_anns,f,cls=NumpyArrayEncoder)
    return stats, coco_evaluator


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def GetAttentionMap(atten,pred,id,outdir):

    pred = pred[-1]
    b,c,w,h = pred.shape

    atten = atten.decompose()[0][0]
    pred = pred[0]

    pred = transforms.ToPILImage()(pred).convert('L')

    atten = transforms.ToPILImage()(atten).convert('L')

    sample_image = Image.new('L',(2*h,w))

    sample_image.paste(pred,(0,0,h,w))
    sample_image.paste(atten,(h,0,2*h,w))


    save_path = os.path.join(outdir,'./save_atten/{}.png'.format(id))
    sample_image.save(save_path)


def plot_bbox_on_img(img,pred,gt,outdir,id):

    b,c,w,h = img.shape


    transform = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = transform(img)
    img = img[0,:,:,:]

    pred_img = transforms.ToPILImage()(img).convert('RGB')

    gt_img = transforms.ToPILImage()(img).convert('RGB')

    draw_pred = ImageDraw.Draw(pred_img)
    draw_gt = ImageDraw.Draw(gt_img)
    for box in pred[0]:

        draw_pred.rectangle(box)

    for box in gt[0]:

        draw_gt.rectangle(box)

    sample_image = Image.new('L',(h,2*w))

    sample_image.paste(pred_img,(0,0,h,w))
    sample_image.paste(gt_img,(0,w,h,2*w))


    save_path = os.path.join(outdir,'./save_bbox/{}.png'.format(id))
    sample_image.save(save_path)

class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor =  F.normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)