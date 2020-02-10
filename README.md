# Yolo-v3 and Yolo-v2 for Windows and Linux
4.  [How to train (Pascal VOC Data)](#how-to-train-pascal-voc-data)
5.  [How to train with multi-GPU:](#how-to-train-with-multi-gpu)
6.  [How to train (to detect your custom objects)](#how-to-train-to-detect-your-custom-objects)
7.  [How to train tiny-yolo (to detect your custom objects)](#how-to-train-tiny-yolo-to-detect-your-custom-objects)
8.  [When should I stop training](#when-should-i-stop-training)
9.  [How to calculate mAP on PascalVOC 2007](#how-to-calculate-map-on-pascalvoc-2007)
10.  [How to improve object detection](#how-to-improve-object-detection)
11.  [How to mark bounded boxes of objects and create annotation files](#how-to-mark-bounded-boxes-of-objects-and-create-annotation-files)
12. [How to use Yolo as DLL and SO libraries](#how-to-use-yolo-as-dll-and-so-libraries)


* Yolo v3 on MS COCO: [Speed / Accuracy (mAP@0.5) chart](https://user-images.githubusercontent.com/4096485/52151356-e5d4a380-2683-11e9-9d7d-ac7bc192c477.jpg)
* Yolo v3 on MS COCO (Yolo v3 vs RetinaNet) - Figure 3: https://arxiv.org/pdf/1804.02767v1.pdf
* Yolo v2 on Pascal VOC 2007: https://hsto.org/files/a24/21e/068/a2421e0689fb43f08584de9d44c2215f.jpg
* Yolo v2 on Pascal VOC 2012 (comp4): https://hsto.org/files/3a6/fdf/b53/3a6fdfb533f34cee9b52bdd9bb0b19d9.jpg

#### Pre-trained models

There are weights-file for different cfg-files (trained for MS COCO dataset):

FPS on GeForce 1080Ti:

* [csresnext50-panet-spp-original-optimal.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/csresnext50-panet-spp-original-optimal.cfg) - **65.4% mAP@0.5 (43.2% AP@0.5..0.95) - 35 FPS** - 100.5 BFlops - 217 MB: [csresnext50-panet-spp-original-optimal_final.weights](https://drive.google.com/open?id=1_NnfVgj0EDtb_WLNoXV8Mo7WKgwdYZCc)

* [yolov3-spp.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-spp.cfg) - **60.6% mAP@0.5 - 30 FPS** - 141.5 BFlops - 240 MB: [yolov3-spp.weights](https://pjreddie.com/media/files/yolov3-spp.weights)

* [yolov3-tiny-prn.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny-prn.cfg) - **33.1% mAP@0.5 - 400 FPS** - 3.5 BFlops - 18.8 MB: [yolov3-tiny-prn.weights](https://drive.google.com/file/d/18yYZWyKbo4XSDVyztmsEcF9B_6bxrhUY/view?usp=sharing)

* [enet-coco.cfg (EfficientNetB0-Yolov3)](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/enet-coco.cfg) - **45.5% mAP@0.5 - 60 FPS** - 3.7 BFlops - 18.3 MB: [enetb0-coco_final.weights](https://drive.google.com/file/d/1FlHeQjWEQVJt0ay1PVsiuuMzmtNyv36m/view)

* [yolov3-openimages.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-openimages.cfg) - 247 MB - OpenImages dataset: [yolov3-openimages.weights](https://pjreddie.com/media/files/yolov3-openimages.weights)

<details><summary><b>CLICK ME</b> - Yolo v3 models</summary>

* [csresnext50-panet-spp.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/csresnext50-panet-spp.cfg) - **60.0% mAP@0.5 - 44 FPS** - 71.3 BFlops - 217 MB: [csresnext50-panet-spp_final.weights](https://drive.google.com/file/d/1aNXdM8qVy11nqTcd2oaVB3mf7ckr258-/view?usp=sharing)

* [yolov3.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg) - **55.3% mAP@0.5 - 46 FPS** - 65.9 BFlops - 236 MB: [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

* [yolov3-tiny.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg) - **33.1% mAP@0.5 - 330 FPS** - 5.6 BFlops - 33.7 MB: [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)

</details>

<details><summary><b>CLICK ME</b> - Yolo v2 models</summary>

* `yolov2.cfg` (194 MB COCO Yolo v2) - requires 4 GB GPU-RAM: https://pjreddie.com/media/files/yolov2.weights
* `yolo-voc.cfg` (194 MB VOC Yolo v2) - requires 4 GB GPU-RAM: http://pjreddie.com/media/files/yolo-voc.weights
* `yolov2-tiny.cfg` (43 MB COCO Yolo v2) - requires 1 GB GPU-RAM: https://pjreddie.com/media/files/yolov2-tiny.weights
* `yolov2-tiny-voc.cfg` (60 MB VOC Yolo v2) - requires 1 GB GPU-RAM: http://pjreddie.com/media/files/yolov2-tiny-voc.weights
* `yolo9000.cfg` (186 MB Yolo9000-model) - requires 4 GB GPU-RAM: http://pjreddie.com/media/files/yolo9000.weights

</details>

Put it near compiled: darknet.exe

You can get cfg-files by path: `darknet/cfg/`

### Requirements

* Windows or Linux
* **CMake >= 3.8** for modern CUDA support: https://cmake.org/download/
* **CUDA 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
* **OpenCV >= 2.4**: use your preferred package manager (brew, apt), build from source using [vcpkg](https://github.com/Microsoft/vcpkg) or download from [OpenCV official site](https://opencv.org/releases.html) (on Windows set system variable `OpenCV_DIR` = `C:\opencv\build` - where are the `include` and `x64` folders [image](https://user-images.githubusercontent.com/4096485/53249516-5130f480-36c9-11e9-8238-a6e82e48c6f2.png))
* **cuDNN >= 7.0 for CUDA 10.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* on Linux **GCC or Clang**, on Windows **MSVC 2015/2017/2019** https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community


### Improvements in this repository

* added State-of-Art models: CSP, PRN, EfficientNet
* added layers: [conv_lstm], [scale_channels] SE/ASFF/BiFPN, [local_avgpool], [sam], [Gaussian_yolo], [reorg3d] (fixed [reorg]), fixed [batchnorm]
* added the ability for training recurrent models (with layers conv-lstm`[conv_lstm]`/conv-rnn`[crnn]`) for accurate detection on video
* added data augmentation: `[net] mixup=1 cutmix=1 mosaic=1 blur=1`. Added activations: SWISH, MISH, NORM_CHAN, NORM_CHAN_SOFTMAX
* added the ability for training with GPU-processing using CPU-RAM to increase the mini_batch_size and increase accuracy (instead of batch-norm sync)
* improved binary neural network performance **2x-4x times** for Detection on CPU and GPU if you trained your own weights by using this XNOR-net model (bit-1 inference) : https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3-tiny_xnor.cfg
* improved neural network performance **~7%** by fusing 2 layers into 1: Convolutional + Batch-norm
* improved performance: Detection **2x times**, on GPU Volta/Turing (Tesla V100, GeForce RTX, ...) using Tensor Cores if `CUDNN_HALF` defined in the `Makefile` or `darknet.sln`
* improved performance **~1.2x** times on FullHD, **~2x** times on 4K, for detection on the video (file/stream) using `darknet detector demo`... 
* improved performance **3.5 X times** of data augmentation for training (using OpenCV SSE/AVX functions instead of hand-written functions) - removes bottleneck for training on multi-GPU or GPU Volta
* improved performance of detection and training on Intel CPU with AVX (Yolo v3 **~85%**)
* optimized memory allocation during network resizing when `random=1`
* optimized GPU initialization for detection - we use batch=1 initially instead of re-init with batch=1
* added correct calculation of **mAP, F1, IoU, Precision-Recall** using command `darknet detector map`...
* added drawing of chart of average-Loss and accuracy-mAP (`-map` flag) during training
* run `./darknet detector demo ... -json_port 8070 -mjpeg_port 8090` as JSON and MJPEG server to get results online over the network by using your soft or Web-browser
* added calculation of anchors for training
* added example of Detection and Tracking objects: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp
* run-time tips and warnings if you use incorrect cfg-file or dataset
* many other fixes of code...

And added manual - [How to train Yolo v3/v2 (to detect your custom objects)](#how-to-train-to-detect-your-custom-objects)

Also, you might be interested in using a simplified repository where is implemented INT8-quantization (+30% speedup and -1% mAP reduced): https://github.com/AlexeyAB/yolo2_light

#### How to use on the command line

On Linux use `./darknet` instead of `darknet.exe`, like this:`./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights`

On Linux find executable file `./darknet` in the root directory, while on Windows find it in the directory `\build\darknet\x64` 

* Yolo v3 COCO - **image**: `darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -thresh 0.25`
* **Output coordinates** of objects: `darknet.exe detector test cfg/coco.data yolov3.cfg yolov3.weights -ext_output dog.jpg`
* Yolo v3 COCO - **video**: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -ext_output test.mp4`
* Yolo v3 COCO - **WebCam 0**: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -c 0`
* Yolo v3 COCO for **net-videocam** - Smart WebCam: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights http://192.168.0.80:8080/video?dummy=param.mjpg`
* Yolo v3 - **save result videofile res.avi**: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights test.mp4 -out_filename res.avi`
* Yolo v3 **Tiny** COCO - video: `darknet.exe detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights test.mp4`
* **JSON and MJPEG server** that allows multiple connections from your soft or Web-browser `ip-address:8070` and 8090: `./darknet detector demo ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights test50.mp4 -json_port 8070 -mjpeg_port 8090 -ext_output`
* Yolo v3 Tiny **on GPU #1**: `darknet.exe detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -i 1 test.mp4`
* Alternative method Yolo v3 COCO - image: `darknet.exe detect cfg/yolov3.cfg yolov3.weights -i 0 -thresh 0.25`
* Train on **Amazon EC2**, to see mAP & Loss-chart using URL like: `http://ec2-35-160-228-91.us-west-2.compute.amazonaws.com:8090` in the Chrome/Firefox (**Darknet should be compiled with OpenCV**): 
    `./darknet detector train cfg/coco.data yolov3.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map`
* 186 MB Yolo9000 - image: `darknet.exe detector test cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights`
* Remeber to put data/9k.tree and data/coco9k.map under the same folder of your app if you use the cpp api to build an app
* To process a list of images `data/train.txt` and save results of detection to `result.json` file use: 
    `darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -ext_output -dont_show -out result.json < data/train.txt`
* To process a list of images `data/train.txt` and save results of detection to `result.txt` use:                             
    `darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -dont_show -ext_output < data/train.txt > result.txt`
* Pseudo-lableing - to process a list of images `data/new_train.txt` and save results of detection in Yolo training format for each image as label `<image_name>.txt` (in this way you can increase the amount of training data) use:
    `darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -thresh 0.25 -dont_show -save_labels < data/new_train.txt`
* To calculate anchors: `darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`
* To check accuracy mAP@IoU=50: `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
* To check accuracy mAP@IoU=75: `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights -iou_thresh 0.75`

##### For using network video-camera mjpeg-stream with any Android smartphone

1. Download for Android phone mjpeg-stream soft: IP Webcam / Smart WebCam

    * Smart WebCam - preferably: https://play.google.com/store/apps/details?id=com.acontech.android.SmartWebCam2
    * IP Webcam: https://play.google.com/store/apps/details?id=com.pas.webcam

2. Connect your Android phone to computer by WiFi (through a WiFi-router) or USB
3. Start Smart WebCam on your phone
4. Replace the address below, on shown in the phone application (Smart WebCam) and launch:

* Yolo v3 COCO-model: `darknet.exe detector demo data/coco.data yolov3.cfg yolov3.weights http://192.168.0.80:8080/video?dummy=param.mjpg -i 0`

### How to compile on Linux (using `cmake`)

The `CMakeLists.txt` will attempt to find installed optional dependencies like
CUDA, cudnn, ZED and build against those. It will also create a shared object
library file to use `darknet` for code development.

Do inside the cloned repository:

```
mkdir build-release
cd build-release
cmake ..
make
make install
```

### How to compile on Linux (using `make`)

Just do `make` in the darknet directory.
Before make, you can set such options in the `Makefile`: [link](https://github.com/AlexeyAB/darknet/blob/9c1b9a2cf6363546c152251be578a21f3c3caec6/Makefile#L1)

* `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
* `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
* `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
* `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
* `DEBUG=1` to bould debug version of Yolo
* `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
* `LIBSO=1` to build a library `darknet.so` and binary runable file `uselib` that uses this library. Or you can try to run so `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib test.mp4` How to use this SO-library from your own code - you can look at C++ example: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp
    or use in such a way: `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov3.cfg yolov3.weights test.mp4`
* `ZED_CAMERA=1` to build a library with ZED-3D-camera support (should be ZED SDK installed), then run
    `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov3.cfg yolov3.weights zed_camera`

To run Darknet on Linux use examples from this article, just use `./darknet` instead of `darknet.exe`, i.e. use this command: `./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights`


## How to train (Pascal VOC Data):

1. Download pre-trained weights for the convolutional layers (154 MB): http://pjreddie.com/media/files/darknet53.conv.74 and put to the directory `build\darknet\x64`

2. Download The Pascal VOC Data and unpack it to directory `build\darknet\x64\data\voc` will be created dir `build\darknet\x64\data\voc\VOCdevkit\`:
    * http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
    * http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    * http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    
    2.1 Download file `voc_label.py` to dir `build\darknet\x64\data\voc`: http://pjreddie.com/media/files/voc_label.py

3. Download and install Python for Windows: https://www.python.org/ftp/python/3.5.2/python-3.5.2-amd64.exe

4. Run command: `python build\darknet\x64\data\voc\voc_label.py` (to generate files: 2007_test.txt, 2007_train.txt, 2007_val.txt, 2012_train.txt, 2012_val.txt)

5. Run command: `type 2007_train.txt 2007_val.txt 2012_*.txt > train.txt`

6. Set `batch=64` and `subdivisions=8` in the file `yolov3-voc.cfg`: [link](https://github.com/AlexeyAB/darknet/blob/ee38c6e1513fb089b35be4ffa692afd9b3f65747/cfg/yolov3-voc.cfg#L3-L4)

7. Start training by using `train_voc.cmd` or by using the command line: 

    `darknet.exe detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74` 

(**Note:** To disable Loss-Window use flag `-dont_show`. If you are using CPU, try `darknet_no_gpu.exe` instead of `darknet.exe`.)

If required change paths in the file `build\darknet\cfg\voc.data`

More information about training by the link: http://pjreddie.com/darknet/yolo/#train-voc

 **Note:** If during training you see `nan` values for `avg` (loss) field - then training goes wrong, but if `nan` is in some other lines - then training goes well.

## How to train with multi-GPU:

1. Train it first on 1 GPU for like 1000 iterations: `darknet.exe detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74`

2. Then stop and by using partially-trained model `/backup/yolov3-voc_1000.weights` run training with multigpu (up to 4 GPUs): `darknet.exe detector train cfg/voc.data cfg/yolov3-voc.cfg /backup/yolov3-voc_1000.weights -gpus 0,1,2,3`

Only for small datasets sometimes better to decrease learning rate, for 4 GPUs set `learning_rate = 0.00025` (i.e. learning_rate = 0.001 / GPUs). In this case also increase 4x times `burn_in =` and `max_batches =` in your cfg-file. I.e. use `burn_in = 4000` instead of `1000`. Same goes for `steps=` if `policy=steps` is set.

https://groups.google.com/d/msg/darknet/NbJqonJBTSY/Te5PfIpuCAAJ

## How to train (to detect your custom objects):
(to train old Yolo v2 `yolov2-voc.cfg`, `yolov2-tiny-voc.cfg`, `yolo-voc.cfg`, `yolo-voc.2.0.cfg`, ... [click by the link](https://github.com/AlexeyAB/darknet/tree/47c7af1cea5bbdedf1184963355e6418cb8b1b4f#how-to-train-pascal-voc-data))

Training Yolo v3:

1. Create file `yolo-obj.cfg` with the same content as in `yolov3.cfg` (or copy `yolov3.cfg` to `yolo-obj.cfg)` and:

  * change line batch to [`batch=64`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L3)
  * change line subdivisions to [`subdivisions=16`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)
  * change line max_batches to (`classes*2000` but not less than `4000`), f.e. [`max_batches=6000`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L20) if you train for 3 classes
  * change line steps to 80% and 90% of max_batches, f.e. [`steps=4800,5400`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L22)
  * set network size `width=416 height=416` or any value multiple of 32: https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L8-L9
  * change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers:
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L610
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L696
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L783
  * change [`filters=255`] to filters=(classes + 5)x3 in the 3 `[convolutional]` before each `[yolo]` layer
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L603
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L689
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L776
  * when using [`[Gaussian_yolo]`](https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L608)  layers, change [`filters=57`] filters=(classes + 9)x3 in the 3 `[convolutional]` before each `[Gaussian_yolo]` layer
      * https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L604
      * https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L696
      * https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L789
      
  So if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`.
  
  **(Do not write in the cfg-file: filters=(classes + 5)x3)**
  
  (Generally `filters` depends on the `classes`, `coords` and number of `mask`s, i.e. filters=`(classes + coords + 1)*<number of mask>`, where `mask` is indices of anchors. If `mask` is absence, then filters=`(classes + coords + 1)*num`)

  So for example, for 2 objects, your file `yolo-obj.cfg` should differ from `yolov3.cfg` in such lines in each of **3** [yolo]-layers:

  ```
  [convolutional]
  filters=21

  [region]
  classes=2
  ```

2. Create file `obj.names` in the directory `build\darknet\x64\data\`, with objects names - each in new line

3. Create file `obj.data` in the directory `build\darknet\x64\data\`, containing (where **classes = number of objects**):

  ```
  classes= 2
  train  = data/train.txt
  valid  = data/test.txt
  names = data/obj.names
  backup = backup/
  ```

4. Put image-files (.jpg) of your objects in the directory `build\darknet\x64\data\obj\`

5. You should label each object on images from your dataset. Use this visual GUI-software for marking bounded boxes of objects and generating annotation files for Yolo v2 & v3: https://github.com/AlexeyAB/Yolo_mark

It will create `.txt`-file for each `.jpg`-image-file - in the same directory and with the same name, but with `.txt`-extension, and put to file: object number and object coordinates on this image, for each object in new line: 

`<object-class> <x_center> <y_center> <width> <height>`

  Where: 
  * `<object-class>` - integer object number from `0` to `(classes-1)`
  * `<x_center> <y_center> <width> <height>` - float values **relative** to width and height of image, it can be equal from `(0.0 to 1.0]`
  * for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
  * atention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner)

  For example for `img1.jpg` you will be created `img1.txt` containing:

  ```
  1 0.716797 0.395833 0.216406 0.147222
  0 0.687109 0.379167 0.255469 0.158333
  1 0.420312 0.395833 0.140625 0.166667
  ```

6. Create file `train.txt` in directory `build\darknet\x64\data\`, with filenames of your images, each filename in new line, with path relative to `darknet.exe`, for example containing:

  ```
  data/obj/img1.jpg
  data/obj/img2.jpg
  data/obj/img3.jpg
  ```

7. Download pre-trained weights for the convolutional layers and put to the directory `build\darknet\x64`
    * for `csresnext50-panet-spp.cfg` (133 MB): [csresnext50-panet-spp.conv.112](https://drive.google.com/file/d/16yMYCLQTY_oDlCIZPfn_sab6KD3zgzGq/view?usp=sharing)
    * for `yolov3.cfg, yolov3-spp.cfg` (154 MB): [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)
    * for `yolov3-tiny-prn.cfg , yolov3-tiny.cfg` (6 MB): [yolov3-tiny.conv.11](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)
    * for `enet-coco.cfg (EfficientNetB0-Yolov3)` (14 MB): [enetb0-coco.conv.132](https://drive.google.com/file/d/1uhh3D6RSn0ekgmsaTcl-ZW53WBaUDo6j/view?usp=sharing)
    

8. Start training by using the command line: `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74`
     
   To train on Linux use command: `./darknet detector train data/obj.data yolo-obj.cfg darknet53.conv.74` (just use `./darknet` instead of `darknet.exe`)
     
   * (file `yolo-obj_last.weights` will be saved to the `build\darknet\x64\backup\` for each 100 iterations)
   * (file `yolo-obj_xxxx.weights` will be saved to the `build\darknet\x64\backup\` for each 1000 iterations)
   * (to disable Loss-Window use `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show`, if you train on computer without monitor like a cloud Amazon EC2)
   * (to see the mAP & Loss-chart during training on remote server without GUI, use command `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map` then open URL `http://ip-address:8090` in Chrome/Firefox browser)

8.1. For training with mAP (mean average precisions) calculation for each 4 Epochs (set `valid=valid.txt` or `train.txt` in `obj.data` file) and run: `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map`

9. After training is complete - get result `yolo-obj_final.weights` from path `build\darknet\x64\backup\`

 * After each 100 iterations you can stop and later start training from this point. For example, after 2000 iterations you can stop training, and later just start training using: `darknet.exe detector train data/obj.data yolo-obj.cfg backup\yolo-obj_2000.weights`

    (in the original repository https://github.com/pjreddie/darknet the weights-file is saved only once every 10 000 iterations `if(iterations > 1000)`)

 * Also you can get result earlier than all 45000 iterations.
 
 **Note:** If during training you see `nan` values for `avg` (loss) field - then training goes wrong, but if `nan` is in some other lines - then training goes well.
 
 **Note:** If you changed width= or height= in your cfg-file, then new width and height must be divisible by 32.
 
 **Note:** After training use such command for detection: `darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`
 
  **Note:** if error `Out of memory` occurs then in `.cfg`-file you should increase `subdivisions=16`, 32 or 64: [link](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)
 
### How to train tiny-yolo (to detect your custom objects):

Do all the same steps as for the full yolo model as described above. With the exception of:
* Download default weights file for yolov3-tiny: https://pjreddie.com/media/files/yolov3-tiny.weights
* Get pre-trained weights `yolov3-tiny.conv.15` using command: `darknet.exe partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15`
* Make your custom model `yolov3-tiny-obj.cfg` based on `cfg/yolov3-tiny_obj.cfg` instead of `yolov3.cfg`
* Start training: `darknet.exe detector train data/obj.data yolov3-tiny-obj.cfg yolov3-tiny.conv.15`

For training Yolo based on other models ([DenseNet201-Yolo](https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/densenet201_yolo.cfg) or [ResNet50-Yolo](https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/resnet50_yolo.cfg)), you can download and get pre-trained weights as showed in this file: https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/partial.cmd
If you made you custom model that isn't based on other models, then you can train it without pre-trained weights, then will be used random initial weights.
 
## When should I stop training:

Usually sufficient 2000 iterations for each class(object), but not less than 4000 iterations in total. But for a more precise definition when you should stop training, use the following manual:

1. During training, you will see varying indicators of error, and you should stop when no longer decreases **0.XXXXXXX avg**:

  > Region Avg IOU: 0.798363, Class: 0.893232, Obj: 0.700808, No Obj: 0.004567, Avg Recall: 1.000000,  count: 8
  > Region Avg IOU: 0.800677, Class: 0.892181, Obj: 0.701590, No Obj: 0.004574, Avg Recall: 1.000000,  count: 8
  >
  > **9002**: 0.211667, **0.60730 avg**, 0.001000 rate, 3.868000 seconds, 576128 images
  > Loaded: 0.000000 seconds

  * **9002** - iteration number (number of batch)
  * **0.60730 avg** - average loss (error) - **the lower, the better**

  When you see that average loss **0.xxxxxx avg** no longer decreases at many iterations then you should stop training. The final avgerage loss can be from `0.05` (for a small model and easy dataset) to `3.0` (for a big model and a difficult dataset).

2. Once training is stopped, you should take some of last `.weights`-files from `darknet\build\darknet\x64\backup` and choose the best of them:

For example, you stopped training after 9000 iterations, but the best result can give one of previous weights (7000, 8000, 9000). It can happen due to overfitting. **Overfitting** - is case when you can detect objects on images from training-dataset, but can't detect objects on any others images. You should get weights from **Early Stopping Point**:

![Overfitting](https://hsto.org/files/5dc/7ae/7fa/5dc7ae7fad9d4e3eb3a484c58bfc1ff5.png) 

To get weights from Early Stopping Point:

  2.1. At first, in your file `obj.data` you must specify the path to the validation dataset `valid = valid.txt` (format of `valid.txt` as in `train.txt`), and if you haven't validation images, just copy `data\train.txt` to `data\valid.txt`.

  2.2 If training is stopped after 9000 iterations, to validate some of previous weights use this commands:

(If you use another GitHub repository, then use `darknet.exe detector recall`... instead of `darknet.exe detector map`...)

* `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
* `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_8000.weights`
* `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_9000.weights`

And comapre last output lines for each weights (7000, 8000, 9000):

Choose weights-file **with the highest mAP (mean average precision)** or IoU (intersect over union)

For example, **bigger mAP** gives weights `yolo-obj_8000.weights` - then **use this weights for detection**.

Or just train with `-map` flag: 

`darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map` 

So you will see mAP-chart (red-line) in the Loss-chart Window. mAP will be calculated for each 4 Epochs using `valid=valid.txt` file that is specified in `obj.data` file (`1 Epoch = images_in_train_txt / batch` iterations)

(to change the max x-axis value - change [`max_batches=`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L20) parameter to `2000*classes`, f.e. `max_batches=6000` for 3 classes)

![loss_chart_map_chart](https://hsto.org/webt/yd/vl/ag/ydvlagutof2zcnjodstgroen8ac.jpeg)

Example of custom object detection: `darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`

* **IoU** (intersect over union) - average instersect over union of objects and detections for a certain threshold = 0.24

* **mAP** (mean average precision) - mean value of `average precisions` for each class, where `average precision` is average value of 11 points on PR-curve for each possible threshold (each probability of detection) for the same class (Precision-Recall in terms of PascalVOC, where Precision=TP/(TP+FP) and Recall=TP/(TP+FN) ), page-11: http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

**mAP** is default metric of precision in the PascalVOC competition, **this is the same as AP50** metric in the MS COCO competition.
In terms of Wiki, indicators Precision and Recall have a slightly different meaning than in the PascalVOC competition, but **IoU always has the same meaning**.

![precision_recall_iou](https://hsto.org/files/ca8/866/d76/ca8866d76fb840228940dbf442a7f06a.jpg)

### How to calculate mAP on PascalVOC 2007:

1. To calculate mAP (mean average precision) on PascalVOC-2007-test:
* Download PascalVOC dataset, install Python 3.x and get file `2007_test.txt` as described here: https://github.com/AlexeyAB/darknet#how-to-train-pascal-voc-data
* Then download file https://raw.githubusercontent.com/AlexeyAB/darknet/master/scripts/voc_label_difficult.py to the dir `build\darknet\x64\data\` then run `voc_label_difficult.py` to get the file `difficult_2007_test.txt`
* Remove symbol `#` from this line to un-comment it: https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/data/voc.data#L4
* Then there are 2 ways to get mAP:
    1. Using Darknet + Python: run the file `build/darknet/x64/calc_mAP_voc_py.cmd` - you will get mAP for `yolo-voc.cfg` model, mAP = 75.9%
    2. Using this fork of Darknet: run the file `build/darknet/x64/calc_mAP.cmd` - you will get mAP for `yolo-voc.cfg` model, mAP = 75.8%
    
 (The article specifies the value of mAP = 76.8% for YOLOv2 416×416, page-4 table-3: https://arxiv.org/pdf/1612.08242v1.pdf. We get values lower - perhaps due to the fact that the model was trained on a slightly different source code than the code on which the detection is was done)

* if you want to get mAP for `tiny-yolo-voc.cfg` model, then un-comment line for tiny-yolo-voc.cfg and comment line for yolo-voc.cfg in the .cmd-file
* if you have Python 2.x instead of Python 3.x, and if you use Darknet+Python-way to get mAP, then in your cmd-file use `reval_voc.py` and `voc_eval.py` instead of `reval_voc_py3.py` and `voc_eval_py3.py` from this directory: https://github.com/AlexeyAB/darknet/tree/master/scripts

### Custom object detection:

Example of custom object detection: `darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`

| ![Yolo_v2_training](https://hsto.org/files/d12/1e7/515/d121e7515f6a4eb694913f10de5f2b61.jpg) | ![Yolo_v2_training](https://hsto.org/files/727/c7e/5e9/727c7e5e99bf4d4aa34027bb6a5e4bab.jpg) |
|---|---|

## How to improve object detection:

1. Before training:
  * set flag `random=1` in your `.cfg`-file - it will increase precision by training Yolo for different resolutions: [link](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L788)

  * increase network resolution in your `.cfg`-file (`height=608`, `width=608` or any value multiple of 32) - it will increase precision


  * check that each object that you want to detect is mandatory labeled in your dataset - no one object in your data set should not be without label. In the most training issues - there are wrong labels in your dataset (got labels by using some conversion script, marked with a third-party tool, ...). Always check your dataset by using: https://github.com/AlexeyAB/Yolo_mark

  * my Loss is very high and mAP is very low, is training wrong? Run training with ` -show_imgs` flag at the end of training command, do you see correct bounded boxes of objects (in windows or in files `aug_...jpg`)? If no - your training dataset is wrong.

  * for each object which you want to detect - there must be at least 1 similar object in the Training dataset with about the same: shape, side of object, relative size, angle of rotation, tilt, illumination. So desirable that your training dataset include images with objects at diffrent: scales, rotations, lightings, from different sides, on different backgrounds - you should preferably have 2000 different images for each class or more, and you should train `2000*classes` iterations or more

  * desirable that your training dataset include images with non-labeled objects that you do not want to detect - negative samples without bounded box (empty `.txt` files) - use as many images of negative samples as there are images with objects

  * What is the best way to mark objects: label only the visible part of the object, or label the visible and overlapped part of the object, or label a little more than the entire object (with a little gap)? Mark as you like - how would you like it to be detected.

  * for training with a large number of objects in each image, add the parameter `max=200` or higher value in the last `[yolo]`-layer or `[region]`-layer in your cfg-file (the global maximum number of objects that can be detected by YoloV3 is `0,0615234375*(width*height)` where are width and height are parameters from `[net]` section in cfg-file) 
  
  * for training for small objects (smaller than 16x16 after the image is resized to 416x416) - set `layers = -1, 11` instead of https://github.com/AlexeyAB/darknet/blob/6390a5a2ab61a0bdf6f1a9a6b4a739c16b36e0d7/cfg/yolov3.cfg#L720
      and set `stride=4` instead of https://github.com/AlexeyAB/darknet/blob/6390a5a2ab61a0bdf6f1a9a6b4a739c16b36e0d7/cfg/yolov3.cfg#L717
  
  * for training for both small and large objects use modified models:
      * Full-model: 5 yolo layers: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3_5l.cfg
      * Tiny-model: 3 yolo layers: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny_3l.cfg
      * Spatial-full-model: 3 yolo layers: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-spp.cfg
  
  * If you train the model to distinguish Left and Right objects as separate classes (left/right hand, left/right-turn on road signs, ...) then for disabling flip data augmentation - add `flip=0` here: https://github.com/AlexeyAB/darknet/blob/3d2d0a7c98dbc8923d9ff705b81ff4f7940ea6ff/cfg/yolov3.cfg#L17
  
  * General rule - your training dataset should include such a set of relative sizes of objects that you want to detect: 

    * `train_network_width * train_obj_width / train_image_width ~= detection_network_width * detection_obj_width / detection_image_width`
    * `train_network_height * train_obj_height / train_image_height ~= detection_network_height * detection_obj_height / detection_image_height`
    
    I.e. for each object from Test dataset there must be at least 1 object in the Training dataset with the same class_id and about the same relative size:

    `object width in percent from Training dataset` ~= `object width in percent from Test dataset` 
   
    That is, if only objects that occupied 80-90% of the image were present in the training set, then the trained network will not be able to detect objects that occupy 1-10% of the image.
    
  * to speedup training (with decreasing detection accuracy) do Fine-Tuning instead of Transfer-Learning, set param `stopbackward=1` here: https://github.com/AlexeyAB/darknet/blob/6d44529cf93211c319813c90e0c1adb34426abe5/cfg/yolov3.cfg#L548
    then do this command: `./darknet partial cfg/yolov3.cfg yolov3.weights yolov3.conv.81 81` will be created file `yolov3.conv.81`,
    then train by using weights file `yolov3.conv.81` instead of `darknet53.conv.74`

  * each: `model of object, side, illimination, scale, each 30 grad` of the turn and inclination angles - these are *different objects* from an internal perspective of the neural network. So the more *different objects* you want to detect, the more complex network model should be used.

  * to make the detected bounded boxes more accurate, you can add 3 parameters `ignore_thresh = .9 iou_normalizer=0.5 iou_loss=giou` to each `[yolo]` layer and train, it will increase mAP@0.9, but decrease mAP@0.5.

  * Only if you are an **expert** in neural detection networks - recalculate anchors for your dataset for `width` and `height` from cfg-file:
  `darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`
   then set the same 9 `anchors` in each of 3 `[yolo]`-layers in your cfg-file. But you should change indexes of anchors `masks=` for each [yolo]-layer, so that 1st-[yolo]-layer has anchors larger than 60x60, 2nd larger than 30x30, 3rd remaining. Also you should change the `filters=(classes + 5)*<number of mask>` before each [yolo]-layer. If many of the calculated anchors do not fit under the appropriate layers - then just try using all the default anchors.


2. After training - for detection:

  * Increase network-resolution by set in your `.cfg`-file (`height=608` and `width=608`) or (`height=832` and `width=832`) or (any value multiple of 32) - this increases the precision and makes it possible to detect small objects: [link](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L8-L9)
  
    * it is not necessary to train the network again, just use `.weights`-file already trained for 416x416 resolution
    * but to get even greater accuracy you should train with higher resolution 608x608 or 832x832, note: if error `Out of memory` occurs then in `.cfg`-file you should increase `subdivisions=16`, 32 or 64: [link](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)

## How to mark bounded boxes of objects and create annotation files:

Here you can find repository with GUI-software for marking bounded boxes of objects and generating annotation files for Yolo v2 & v3: https://github.com/AlexeyAB/Yolo_mark

With example of: `train.txt`, `obj.names`, `obj.data`, `yolo-obj.cfg`, `air`1-6`.txt`, `bird`1-4`.txt` for 2 classes of objects (air, bird) and `train_obj.cmd` with example how to train this image-set with Yolo v2 & v3

## Using Yolo9000

 Simultaneous detection and classification of 9000 objects: `darknet.exe detector test cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights data/dog.jpg`

* `yolo9000.weights` - (186 MB Yolo9000 Model) requires 4 GB GPU-RAM: http://pjreddie.com/media/files/yolo9000.weights

* `yolo9000.cfg` - cfg-file of the Yolo9000, also there are paths to the `9k.tree` and `coco9k.map`  https://github.com/AlexeyAB/darknet/blob/617cf313ccb1fe005db3f7d88dec04a04bd97cc2/cfg/yolo9000.cfg#L217-L218

    * `9k.tree` - **WordTree** of 9418 categories  - `<label> <parent_it>`, if `parent_id == -1` then this label hasn't parent: https://raw.githubusercontent.com/AlexeyAB/darknet/master/build/darknet/x64/data/9k.tree

    * `coco9k.map` - map 80 categories from MSCOCO to WordTree `9k.tree`: https://raw.githubusercontent.com/AlexeyAB/darknet/master/build/darknet/x64/data/coco9k.map

* `combine9k.data` - data file, there are paths to: `9k.labels`, `9k.names`, `inet9k.map`, (change path to your `combine9k.train.list`): https://raw.githubusercontent.com/AlexeyAB/darknet/master/build/darknet/x64/data/combine9k.data

    * `9k.labels` - 9418 labels of objects: https://raw.githubusercontent.com/AlexeyAB/darknet/master/build/darknet/x64/data/9k.labels

    * `9k.names` -
9418 names of objects: https://raw.githubusercontent.com/AlexeyAB/darknet/master/build/darknet/x64/data/9k.names

    * `inet9k.map` - map 200 categories from ImageNet to WordTree `9k.tree`: https://raw.githubusercontent.com/AlexeyAB/darknet/master/build/darknet/x64/data/inet9k.map


## How to use Yolo as DLL and SO libraries

* using `build.sh` or
* build `darknet` using `cmake` or
* set `LIBSO=1` in the `Makefile` and do `make`

There are 2 APIs:
* C API: https://github.com/AlexeyAB/darknet/blob/master/include/darknet.h
    * Python examples using the C API::     
         * https://github.com/AlexeyAB/darknet/blob/master/darknet.py	
         * https://github.com/AlexeyAB/darknet/blob/master/darknet_video.py
    
* C++ API: https://github.com/AlexeyAB/darknet/blob/master/include/yolo_v2_class.hpp
    * C++ example that uses C++ API: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp
    
----

1. To compile Yolo as C++ DLL-file `yolo_cpp_dll.dll` - open the solution `build\darknet\yolo_cpp_dll.sln`, set **x64** and **Release**, and do the: Build -> Build yolo_cpp_dll
    * You should have installed **CUDA 10.0**
    * To use cuDNN do: (right click on project) -> properties -> C/C++ -> Preprocessor -> Preprocessor Definitions, and add at the beginning of line: `CUDNN;`

2. To use Yolo as DLL-file in your C++ console application - open the solution `build\darknet\yolo_console_dll.sln`, set **x64** and **Release**, and do the: Build -> Build yolo_console_dll

    * you can run your console application from Windows Explorer `build\darknet\x64\yolo_console_dll.exe`
    **use this command**: `yolo_console_dll.exe data/coco.names yolov3.cfg yolov3.weights test.mp4`
    
    * after launching your console application and entering the image file name - you will see info for each object: 
    `<obj_id> <left_x> <top_y> <width> <height> <probability>`
    * to use simple OpenCV-GUI you should uncomment line `//#define OPENCV` in `yolo_console_dll.cpp`-file: [link](https://github.com/AlexeyAB/darknet/blob/a6cbaeecde40f91ddc3ea09aa26a03ab5bbf8ba8/src/yolo_console_dll.cpp#L5)
    * you can see source code of simple example for detection on the video file: [link](https://github.com/AlexeyAB/darknet/blob/ab1c5f9e57b4175f29a6ef39e7e68987d3e98704/src/yolo_console_dll.cpp#L75)
   
`yolo_cpp_dll.dll`-API: [link](https://github.com/AlexeyAB/darknet/blob/master/src/yolo_v2_class.hpp#L42)
```
struct bbox_t {
    unsigned int x, y, w, h;    // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;        // class of object - from range [0, classes-1]
    unsigned int track_id;        // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter;// counter of frames on which the object was detected
};

class Detector {
public:
        Detector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0);
        ~Detector();

        std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2, bool use_mean = false);
        std::vector<bbox_t> detect(image_t img, float thresh = 0.2, bool use_mean = false);
        static image_t load_image(std::string image_filename);
        static void free_image(image_t m);

#ifdef OPENCV
        std::vector<bbox_t> detect(cv::Mat mat, float thresh = 0.2, bool use_mean = false);
	std::shared_ptr<image_t> mat_to_image_resize(cv::Mat mat) const;
#endif
};
```
