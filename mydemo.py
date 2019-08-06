import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import xml.etree.ElementTree as ET
from dataloader import crop_from_dets
from SPPE.src.utils.img import im_to_torch

from pPose_nms import pose_nms, write_json

import cv2

args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    if len(inputlist):
        im_names = open(inputlist, 'r').readlines()
    elif len(inputpath) and inputpath != '/':
        for root, dirs, files in os.walk(inputpath):
            im_names = files
    else:
        raise IOError('Error: must contain either --indir/--list')

    # Load input images
    data_loader = ImageLoader(im_names, batchSize=args.detbatch, format='yolo').start()

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
  
    #det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    #det_processor = DetectionProcessor(det_loader).start()
    xml_root_path = './test/xml/'
    
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    writer = DataWriter(args.save_video).start()

    data_len = data_loader.length()
    im_names_desc = tqdm(range(data_len))

    batchSize = args.posebatch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            #(inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            # revise by dengyang
            boxes = []
            scores = []
            img, orig_img, im_name, im_dim_list = data_loader.getitem()
            print (im_name)
            path, filename = os.path.split(im_name[0])
            name, extension = os.path.splitext(filename)
            xml_file = name + '.xml'
            xml_path = os.path.join(xml_root_path, xml_file)
            # parse xml to get bbox
            xmlfilepath = os.path.abspath(xml_path)
            tree = ET.parse(xmlfilepath)
            root = tree.getroot()
            for member in root.findall('object'):
                boxes.append([float(member[4][0].text), float(member[4][1].text),
                             float(member[4][2].text), float(member[4][3].text),])
                scores.append([1.])
            print (boxes)
            # input 
            inps = torch.zeros(len(boxes), 3, opt.inputResH, opt.inputResW)
            pt1 = torch.zeros(len(boxes), 2)
            pt2 = torch.zeros(len(boxes), 2)
            boxes = torch.Tensor(boxes)
            scores = torch.Tensor(scores)
            orig_img = orig_img[0]
            inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)
            print (inps.shape)

            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation
            
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)
            hm = hm.cpu()
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name[0].split('/')[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)
        
        if args.profile:
            # TQDM
            im_names_desc.set_description(
            'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )

    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    write_json(final_result, args.outputpath)
