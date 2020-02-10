#! /usr/bin/python
from __future__ import division

import argparse
import glob
import os
import sys
import time
import uuid
from ctypes import *
from random import shuffle

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append('/workspace/darknet')
import darknet as dn


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--thresh', type=float, default=.4)
parser.add_argument('--imsize', type=int, default=576)
parser.add_argument('--preproc', action="store_true")
parser.add_argument('--weights_folder', type=str, default='/workspace/darknet/backup')
parser.add_argument('--model', type=str, default='/workspace/darknet/cfg/yolo_watermark.cfg')
parser.add_argument('--watermarked_images_path', type=str, default='/workspace/darknet/data/logos/images')
parser.add_argument('--meta', type=str, default='/workspace/darknet/cfg/watermark.data')
parser.add_argument('--labels_path', type=str, default='/workspace/darknet/data/logos/labels/')
parser.add_argument('--labels_file', type=str, default='/workspace/darknet/data/logos/labels.txt')
parser.add_argument('--border', type=int, default=20)
args = parser.parse_args()

dn.set_gpu(args.gpu)
metadata = dn.load_meta(args.meta.encode())
labels = open(args.labels_file, 'r').read().splitlines()


def preprocess(img, border=20):
    # resize
    img = img.resize((args.imsize, args.imsize), resample=Image.BILINEAR)

    # border
    img = np.asarray(img)
    img = cv2.rectangle(img, (0, 0), img.shape[:2][::-1], (255, 255, 255), border)
    img = Image.fromarray(img)

    return img


def detect(net, meta, filename, thresh=.5, hier_thresh=.5, nms=.45, preproc=True):
    """
    if preproc:
        img = Image.open(filename)
        img = preprocess(img, args.border)
        filename = '/tmp/' + str(uuid.uuid4()) + '.tiff'
        img.save(filename, format='TIFF')
        img = dn.load_image(filename.encode(), 0, 0)
        os.remove(filename)
    else:
        img = dn.load_image(filename.encode(), 0, 0)

    #boxes = dn.make_network_boxes(net)
    #probs = dn.make_probs(net)
    #num =   dn.num_boxes(net)
    #dn.network_detect(net, img, thresh, hier_thresh, nms, boxes, probs)
    """
    res = []
    result = dn.detect(net,metadata, filename.encode("ascii"), thresh, hier_thresh, nms, False)
    """
     for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    """
    #dn.free_image(img)
    #dn.free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return result


def cmp(t, p):
    return (t and p) or (not t and not p)


def main():

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    best = ""
    best_precision = 0
    avg_time = -1
    images = glob.glob(args.watermarked_images_path + '/*jpg')
    #images = open(args.test_file, 'r').read().splitlines()
    shuffle(images)
    print('{} images loaded'.format(len(images)))
    models = glob.glob(args.weights_folder + '/*.weights')
    print('Testing {} models'.format(len(models)))
    for i in models:
        #pbar = tqdm(total=len(images))
        print(args.model)
        print(i)
        net = dn.load_net(args.model.encode(), i.encode(), 0)
        #dn.set_batch_network(net, 1)
        #dn.resize_network(net, args.imsize, args.imsize)
        sys.stderr.write('Resized to %d x %d\n' % (dn.network_width(net), dn.network_height(net)))
        for count, img in enumerate(images, 1):
            start = time.time()
            res = detect(net, metadata, img, thresh=args.thresh, preproc=args.preproc)
            end = time.time()
            avg_time = (end-start) if avg_time < 0 else avg_time*.9 + (end-start)*.1
            p = map(lambda x: x[0], res)
            t = map(lambda x: labels[int(x.split(' ')[0])], open(args.labels_path + img.rsplit('/', 1)[1].split('.')[0] + '.txt', 'r').read().splitlines())
            if t and p:
                tp += 1
            if not t and p:
                fp += 1
            if not t and not p:
                tn += 1
            if t and not p:
                fn += 1
            precision = tp / (tp+fp) if tp+fp > 0 else 0
            recall = tp / (tp+fn) if tp+fn > 0 else 0
            specificity = tn / (tn+fp) if tn+fp > 0 else 0
            #pbar.update(1)
        #pbar.close()
        print('Using model: {}'.format(i))
        sys.stdout.write('\rPrecision: %.4f, Recall: %.4f, Specificity: %.4f, %.4f seconds, %d/%d images' % (precision, recall, specificity, avg_time, count, len(images)))
        sys.stdout.flush()
        if precision > best_precision:
            best = i
            best_precision = precision

    print('Best model: {}'.format(best))
    print('Best precision: {}'.format(best_precision))

if __name__ == '__main__':
    main()


