#! /usr/bin/python
from __future__ import division

import os
import sys
import argparse
import warnings

import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.append('/workspace/darknet')
import darknet as dn

warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--thresh', type=float, default=.001)
	parser.add_argument('--iou_thresh', type=float, default=.01)
	parser.add_argument('--imsize', type=int, default=608)
	parser.add_argument('--meta_data', type=str)
	parser.add_argument('--model', type=str,
						default='/srv/data/re_logo/published_12/deploy.cfg')
	parser.add_argument('--weights', type=str,
						default='/srv/data/re_logo/published_12/trained_model.weights')
	parser.add_argument('--labels', type=str,
						default='/srv/data/re_logo/published_12/corresp.names')
	parser.add_argument('--test_file', type=str,
						default='/home/case/darknet/data/re_logo_v12/val.txt')
	parser.add_argument('--results_file', type=str)
	return parser.parse_args()


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


def parse_annotations_file(fp):
	lines = open(fp, 'r').read().splitlines()
	annotations = [line.split(' ') for line in lines]
	true = []
	for ann in annotations:
		cls = int(ann[0])
		bbox = tuple(map(float, ann[1:]))
		true.append((cls, bbox))
	return true


def init_net(args):
	dn.set_gpu(args.gpu)
	net = dn.load_net(args.model.encode(), args.weights.encode(), 0)

	dn.set_batch_network(net, 1)
	dn.resize_network(net, args.imsize, args.imsize)
	sys.stderr.write('Resized to %d x %d\n' %
					 (dn.network_width(net), dn.network_height(net)))

	return net


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
	"""
	From darknet/python/darknet.py
	"""
	# im = dn.load_image(image, 0, 0)
	im = dn.load_image(image, dn.network_width(net), dn.network_height(net))
	boxes = dn.make_boxes(net)
	probs = dn.make_probs(net)
	num = dn.num_boxes(net)
	dn.network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
	res = []
	for j in range(num):
		for i in range(meta.classes):
			if probs[j][i] > 0:
				res.append(
					(i, probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
	res = sorted(res, key=lambda x: -x[1])
	dn.free_image(im)
	dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
	return res


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
	#print('BEFORE:\n BoxA:{}\nBoxB:{}'.format(boxA,boxB))
	boxA = (
		np.clip(np.clip(boxA[0], 0, 1) - boxA[2] / 2, 0, 1),  # left
		np.clip(np.clip(boxA[1], 0, 1) - boxA[3] / 2, 0, 1),  # top
		np.clip(np.clip(boxA[0], 0, 1) + boxA[2] / 2, 0, 1),  # right
		np.clip(np.clip(boxA[1], 0, 1) + boxA[3] / 2, 0, 1)   # bottom
	)
	boxB = (
		np.clip(np.clip(boxB[0], 0, 1) - boxB[2] / 2, 0, 1),  # left
		np.clip(np.clip(boxB[1], 0, 1) - boxB[3] / 2, 0, 1),  # top
		np.clip(np.clip(boxB[0], 0, 1) + boxB[2] / 2, 0, 1),  # right
		np.clip(np.clip(boxB[1], 0, 1) + boxB[3] / 2, 0, 1)   # bottom
	)

	#print('AFTER:\n BoxA:{}\nBoxB:{}'.format(boxA,boxB))

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
	#print('Intersection area:{}'.format(inter_area))
	#print('Union area:{}'.format(float(boxA_area + boxB_area - inter_area)))
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
	tp = [0] * nlabels	# true positive
	fp = [0] * nlabels	# false positive
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
					#print('Computing overlap interesection')
					# compute overlap as area of intersection / area of union
					iou = bbox_iou(det.bbox, ann.bbox)
					#print('IoU:{}'.format(iou))
					if iou > ovmax:
						ovmax = iou
						jmax = j
			#print('Computed overlap:{}'.format(ovmax))
			# Assign detection as true positive/don't care/false positive
			if ovmax > iou_thresh:
				if not detected[jmax]:
					tp[det.label] += 1	# true positive
					detected[jmax] = True
				else:
					# fp[det.label] += 1  # false positive (multiple detection)
					multi_det[det.label] += 1
			else:
				fp[det.label] += 1	# false positive
				for ann in filter(lambda ann: ann.label != det.label, gt):
					if bbox_iou(det.bbox, ann.bbox) > iou_thresh:
						wrong_cls[det.label] += 1
						break
				else:
					low_iou[det.label] += 1
	#print('Object false positive due to overlap below thresholds: {}'.format(sum(low_iou)))
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
	tp = [0] * nlabels	# true positive
	fp = [0] * nlabels	# false positive
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


def print_setup(args):
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
	print('')


def _resize_image(img, w, h):
	im_w, im_h = img.size
	if w/im_w < h/im_h:
		new_w = w
		new_h = (im_h * w) // im_w
	else:
		new_h = h
		new_w = (im_w * h) // im_h
 
	resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
	return resized


def main():
	args = parse_args()
	print_setup(args)

	with open(args.labels, 'r') as f:
		labels = f.read().splitlines()

	net = init_net(args)

	print('\nTesting images from {}...'.format(args.test_file))
	images = open(args.test_file, 'r').read().splitlines()
	#images = np.random.choice(images, 10)

	pbar = tqdm(total=len(images))

	# Predict images and load ground truth
	predicted, actual = [], []
	for num, image in enumerate(images):
		img = Image.open(image)
		arr = np.array(img)
		# Predict image
		#pred = dn.detect(net, Metadata(len(labels), labels), img, thresh=args.thresh)
		#dn.free_image(img)
		pred = dn.detect(net, Metadata(len(labels), labels), image.encode(), thresh=args.thresh)
	# pred = [Detection(det[0], det[1], det[2]) for det in pred]
		# toca normalitzar el tamany
		pred_aux = list()
		for det in pred:
			normalized = [float(det[2][0])/arr.shape[1], float(det[2][1])/arr.shape[0], float(det[2][2])/arr.shape[1], float(det[2][3])/arr.shape[0]]
			pred_aux.append(Detection(0, det[1], normalized))   # independent of label
		# Load ground truth
		annotations_file = image.replace('images', 'labels', 1).rsplit('.', 1)[0] + '.txt'
		gt = parse_annotations_file(annotations_file)
		# gt = [Annotation(ann[0], ann[1]) for ann in gt]
		gt = [Annotation(0, ann[1]) for ann in gt]	# independent of label

		# pred = postprocess_kitchen_features(pred)
		# gt = postprocess_kitchen_features(gt)

		predicted.append(pred_aux)
		actual.append(gt)
		pbar.update(1)
	pbar.close()
	"""
		results = dn.performDetect(image, 0.25, args.model, args.weights, args.meta_data, True, True, False)
		print(results)
		print(results['image'])
		print(type(results['image']))
		Image.fromarray(results['image']).save('./test/'+str(num)+'.jpg')
	# Save detections
	if args.results_file:
		with open(args.results_file, 'w') as f:
			for image, pred in zip(images, predicted):
				name = os.path.splitext(os.path.split(image)[1])[0]
				with Image.open(image) as img:
					im_w, im_h = img.size
				for det in pred:
					x, y, w, h = det.bbox

					xmin = x - w/2 + 1
					xmax = x + w/2 + 1
					ymin = y - h/2 + 1
					ymax = y + h/2 + 1

					if xmin < 1: xmin = 1
					if ymin < 1: ymin = 1
					if xmax > im_w: xmax = w
					if ymax > im_h: ymax = h

					#f.write('{} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(name, labels[det.label], det.confidence, xmin, ymin, xmax, ymax))
	"""
	print('\nResults')
	print('-------')
	for thresh in np.arange(0.05, 1, 0.05):
		# Compute evaluation metrics
		tp, fp, npos = compute_metrics(predicted, actual, len(labels), thresh, args.iou_thresh)
		#tp, fp, npos, nneg = compute_metrics_without_iou(predicted, actual, len(labels), thresh)

		prec = sum(tp) / (sum(tp) + sum(fp)) if (sum(tp) + sum(fp)) > 0 else 0
		rec = sum(tp) / sum(npos) if sum(npos) > 0 else 0
		#spc = 1 - sum(fp) / sum(nneg) if sum(nneg) > 0 else 0

		print('Threshold: {:.2f}, Precision: {:.4f}, Recall: {:.4f}'.format(thresh, prec, rec))
		#print('Threshold: {:.2f}, Precision: {:.4f}, Specificity: {:.4f}, Recall: {:.4f}'.format(thresh, prec, spc, rec))

		print_results_per_class = True
		if print_results_per_class:
			for i in range(len(labels)):
				prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
				rec = tp[i] / npos[i] if npos[i] > 0 else 0
				#spc = 1 - fp[i] / nneg[i] if nneg[i] > 0 else 0

				print('  {}, {:.4f}, {:.4f}'.format(labels[i], prec, rec))
				#print('  {}, {:.4f}, {:.4f}, {:.4f}'.format(labels[i], prec, rec, spc))
				# print('  {}, {:.4f}, {:.4f}'.format(labels[i], spc, rec))

if __name__ == '__main__':
	main()

