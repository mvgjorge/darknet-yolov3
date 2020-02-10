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
parser.add_argument('--weight_precision', type=float, default=0.5,
					help='Percentage that will be used while scoring a model: precision * PERCENTAGE + recall * (1- PRECENTAGE)')
parser.add_argument('--labels_path', type=str, default='/workspace/darknet/data/logos/labels/')
parser.add_argument('--print_results_per_class', type=bool, default=False)
parser.add_argument('--labels_file', type=str, default='/workspace/darknet/data/logos/labels.txt')
parser.add_argument('--border', type=int, default=20)
args = parser.parse_args()

dn.set_gpu(args.gpu)
metadata = dn.load_meta(args.meta.encode())
labels = open(args.labels_file, 'r').read().splitlines()


class Metadata:
	def __init__(self, classes, names):
		self.classes = classes
		self.names = names


class Annotation:
	def __init__(self, label, bbox):
		self.label = label
		self.bbox = bbox


class Detection:
	def __init__(self, label, confidence, bbox):
		self.label = label
		self.confidence = confidence
		self.bbox = bbox


def preprocess(img, border=20):
	# resize
	img = img.resize((args.imsize, args.imsize), resample=Image.BILINEAR)

	# border
	img = np.asarray(img)
	img = cv2.rectangle(img, (0, 0), img.shape[:2][::-1], (255, 255, 255), border)
	img = Image.fromarray(img)

	return img


def init_net(weights):
	dn.set_gpu(args.gpu)
	net = dn.load_net(args.model.encode(), weights, 0)

	dn.set_batch_network(net, 1)
	dn.resize_network(net, args.imsize, args.imsize)
	sys.stderr.write('Resized to %d x %d\n' % (dn.network_width(net), dn.network_height(net)))
	return net


def bbox_iou(boxA, boxB):
	"""
	Computes the overlap between two bounding boxes.

	Args:
		boxA: A sequence of floats representing a bounding box in (cx,cy,w,h)
			format with relative coordinates (between 0 and 1).
		boxB: A sequence of floats representing a bounding box in (cx,cy,w,h)
			format with relative coordinates (between 0 and 1).

	Returns:
		The interesection over union between the bounding boxes as a percentage
		(between 0 and 1).
	"""

	# convert from (cx,cy,w,h) to (x1,y1,x2,y2)
	boxA = (
		np.clip(np.clip(boxA[0], 0, 1) - boxA[2] / 2, 0, 1),  # left
		np.clip(np.clip(boxA[1], 0, 1) - boxA[3] / 2, 0, 1),  # top
		np.clip(np.clip(boxA[0], 0, 1) + boxA[2] / 2, 0, 1),  # right
		np.clip(np.clip(boxA[1], 0, 1) + boxA[3] / 2, 0, 1)  # bottom
	)
	boxB = (
		np.clip(np.clip(boxB[0], 0, 1) - boxB[2] / 2, 0, 1),  # left
		np.clip(np.clip(boxB[1], 0, 1) - boxB[3] / 2, 0, 1),  # top
		np.clip(np.clip(boxB[0], 0, 1) + boxB[2] / 2, 0, 1),  # right
		np.clip(np.clip(boxB[1], 0, 1) + boxB[3] / 2, 0, 1)  # bottom
	)

	assert boxA[0] < boxA[2]
	assert boxA[1] < boxA[3]
	assert boxB[0] < boxB[2]
	assert boxB[1] < boxB[3]

	# determine the (x, y)-coordinates of the intersection rectangle
	x_left = max(boxA[0], boxB[0])
	y_top = max(boxA[1], boxB[1])
	x_right = min(boxA[2], boxB[2])
	y_bottom = min(boxA[3], boxB[3])

	# no intersection
	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# compute the area of intersection rectangle
	inter_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = inter_area / float(boxA_area + boxB_area - inter_area)
	# print('Intersection area:{}'.format(inter_area))
	# print('Union area:{}'.format(float(boxA_area + boxB_area - inter_area)))
	assert iou >= 0.0
	assert iou <= 1.0

	# return the intersection over union value
	return iou


def compute_metrics(predicted, actual, nlabels, thresh, iou_thresh):
	"""
	Computes TP, FP and NUM_POS taking into account intersection over union of predictions and annotations.

	:param predicted: List of image predictions.
	:param actual: List of image annotations.
	:param nlabels: Number of labels.
	:param thresh: Confidence threshold.
	:param iou_thresh: Overlap threshold.
	:return: TP, FP, NUM_POS
	"""
	tp = [0] * nlabels  # true positive
	fp = [0] * nlabels  # false positive
	npos = [0] * nlabels  # condition positive

	multi_det = [0] * nlabels  # object with multiple detections
	low_iou = [0] * nlabels  # false positive due to overlap below threshold
	# false positive due to wrong class with overlap above threshold
	wrong_cls = [0] * nlabels

	for pred, gt in zip(predicted, actual):
		# Compute num positives per class
		for ann in gt:
			npos[ann.label] += 1

		# Sort detections by decreasing confidence
		pred.sort(key=lambda det: det.confidence, reverse=True)
		# Assign detections to ground truth objects
		detected = [False] * len(gt)
		for det in pred:
			# discard detections with low confidence
			if det.confidence < thresh:
				continue

			# Assign detection to ground truth object if any
			ovmax = 0
			for j, ann in enumerate(gt):
				if ann.label == det.label:
					# print('Computing overlap interesection')
					# compute overlap as area of intersection / area of union
					iou = bbox_iou(det.bbox, ann.bbox)
					# print('IoU:{}'.format(iou))
					if iou > ovmax:
						ovmax = iou
						jmax = j
			# print('Computed overlap:{}'.format(ovmax))
			# Assign detection as true positive/don't care/false positive
			if ovmax > iou_thresh:
				if not detected[jmax]:
					tp[det.label] += 1  # true positive
					detected[jmax] = True
				else:
					# fp[det.label] += 1  # false positive (multiple detection)
					multi_det[det.label] += 1
			else:
				fp[det.label] += 1  # false positive
				for ann in filter(lambda ann: ann.label != det.label, gt):
					if bbox_iou(det.bbox, ann.bbox) > iou_thresh:
						wrong_cls[det.label] += 1
						break
				else:
					low_iou[det.label] += 1
	# print('Object false positive due to overlap below thresholds: {}'.format(sum(low_iou)))
	# return tp, fp, npos, multi_det, low_iou, wrong_cls
	return tp, fp, npos


def compute_metrics_without_iou(predicted, actual, nlabels, thresh):
	"""
	Computes TP, FP, NUM_POS and NUM_NEG without taking into account bbox positions.

	:param predicted: List of image predictions.
	:param actual: List of image annotations.
	:param nlabels: Number of labels.
	:param thresh: Confidence threshold.
	:return: TP, FP, NUM_POS, NUM_NEG
	"""
	tp = [0] * nlabels  # true positive
	fp = [0] * nlabels  # false positive
	npos = [0] * nlabels  # condition positive
	nneg = [0] * nlabels  # condition negative

	for pred, gt in zip(predicted, actual):
		# count prediction labels
		pred_labels = [0] * nlabels
		for det in pred:
			if det.confidence < thresh:
				continue
			pred_labels[det.label] += 1

		# count ground truth labels
		gt_labels = [0] * nlabels
		for ann in gt:
			gt_labels[ann.label] += 1

		for i in range(nlabels):
			if gt_labels[i] >= 1:
				npos[i] += 1
				if pred_labels[i] >= 1:
					tp[i] += 1
			else:
				nneg[i] += 1
				if pred_labels[i] >= 1:
					fp[i] += 1

	return tp, fp, npos, nneg


def print_setup():
	print('\nExperimental setup')
	print('------------------')
	print('GPU: {}'.format(args.gpu))
	print('Confidence threshold: {}'.format(args.thresh))
	print('IoU threshold: {}'.format(args.iou_thresh))
	print('Input size: {}'.format(args.imsize))
	print('Model: {}'.format(args.model))
	print('Weights: {}'.format(args.weights))
	print('Labels: {}'.format(args.labels))
	print('Test file: {}'.format(args.test_file))
	print('Results file: {}'.format(args.results_file))
	print(''


def main():
	print_setup()
	with open(args.labels, 'r') as f:
		labels = f.read().splitlines()

	print('\nTesting images from {}...'.format(args.test_file))
	images = open(args.test_file, 'r').read().splitlines()
	images = np.random.choice(images, 200)
	shuffle(images)
	print('{} images loaded'.format(len(images)))

	models = glob.glob(args.weights_folder + '/*.weights')
	print('Testing {} models'.format(len(models)))
	pbar = tqdm(total=len(models))
	best_score, best_precision, best_recall, threshold = 0, 0, 0, 0
	for i in models:
		print(args.model)
		net = init_net(i.encode())

		sys.stderr.write('Resized to %d x %d\n' % (dn.network_width(net), dn.network_height(net)))
		# Predict images and load ground truth
		predicted, actual = [], []

		for count, img in enumerate(images):
			start = time.time()
			img = Image.open(image)
			arr = np.array(img)
			pred = dn.detect(net, Metadata(len(labels), labels), image.encode(), thresh=args.thresh)
			pred_aux = list()
			for det in pred:
				# normalizing bounding box points
				normalized = [float(det[2][0]) / arr.shape[1], float(det[2][1]) / arr.shape[0],
							  float(det[2][2]) / arr.shape[1], float(det[2][3]) / arr.shape[0]]
				pred_aux.append(Detection(0, det[1], normalized))  # independent of label
			# Load ground truth
			annotations_file = image.replace('images', 'labels', 1).rsplit('.', 1)[0] + '.txt'
			gt = parse_annotations_file(annotations_file)
			# gt = [Annotation(ann[0], ann[1]) for ann in gt]
			gt = [Annotation(0, ann[1]) for ann in gt]  # independent of label
			predicted.append(pred_aux)
			actual.append(gt)
			end = time.time()
			avg_time = (end - start) if avg_time < 0 else avg_time * .9 + (end - start) * .1
		pbar.update(1)
		score_inside_model = 0
		precision_inside_model = 0
		recall_inside_model = 0
		threshold_inside_model = 0
		for thresh in np.arange(0.05, 1, 0.05):
			# Compute evaluation metrics
			tp, fp, npos = compute_metrics(predicted, actual, len(labels), thresh, args.iou_thresh)
			# tp, fp, npos, nneg = compute_metrics_without_iou(predicted, actual, len(labels), thresh)

			prec = sum(tp) / (sum(tp) + sum(fp)) if (sum(tp) + sum(fp)) > 0 else 0
			rec = sum(tp) / sum(npos) if sum(npos) > 0 else 0
			# spc = 1 - sum(fp) / sum(nneg) if sum(nneg) > 0 else 0

			if args.print_results_per_class:
				for j in range(len(labels)):
					prec = tp[j] / (tp[j] + fp[j]) if (tp[j] + fp[j]) > 0 else 0
					rec = tp[j] / npos[j] if npos[j] > 0 else 0
					# spc = 1 - fp[j] / nneg[j] if nneg[j] > 0 else 0

			total_score = prec * args.weight_precision + rec * (1 - args.weight_accuracy)
			if total_score > score_inside_model:
				score_inside_model = total_score
				threshold_inside_model = thresh
				precision_inside_model = prec
				recall_inside_model = rec

		sys.stdout.write('\rPrecision: %.4f, Recall: %.4f, Threshold: %.4f,  %.4f seconds, %d/%d images' % (
		precision_inside_model, recall_inside_model, threshold_inside_model, avg_time, count, len(images)))
		sys.stdout.flush()
		if score_inside_model > best_score:
			best_score = score_inside_model
			best_precision = precision_inside_model
			best_recall =
			threshold = threshold_inside_model
			best = i
	pbar.close()

	print('Best model: {}'.format(best))
	print('Best precision, recall and threshold: {}, {}, {}'.format(best_precision, best_recall, threshold))


if __name__ == '__main__':
	main()
