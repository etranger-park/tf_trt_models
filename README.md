TensorFlow/TensorRT Models on Jetson
====================================

<p align="center">
<img src="data/landing_graphic.jpg" alt="landing graphic" height="300px"/>
</p>

This repository contains scripts and documentation to use TensorFlow image classification and object detection models on NVIDIA Jetson.  The models are sourced from the [TensorFlow models repository](https://github.com/tensorflow/models)
and optimized using TensorRT.

- [TensorFlow/TensorRT Models on Jetson](#tensorflowtensorrt-models-on-jetson)
  - [Setup](#setup)
  - [Object Detection](#object-detection)
    - [Models](#models)
    - [Download pretrained model](#download-pretrained-model)
    - [Build TensorRT / Jetson compatible graph](#build-tensorrt--jetson-compatible-graph)
    - [Optimize with TensorRT](#optimize-with-tensorrt)
    - [Structure](#structure)
  - [Run](#run)

<a name="setup"></a>
Setup
-----

1. Flash your Jetson TX2 with JetPack 3.2 (including TensorRT).
2. Install miscellaneous dependencies on Jetson

   ```
   sudo apt-get install python-pip python-matplotlib python-pil
   ```
   
3. If your using tensorRT 3, Install TensorFlow 1.7 ~ 1.10 (with TensorRT support). If your using tensorRT 4, Install TensorFlow 1.11 +  Download the [pre-built pip wheel](https://devtalk.nvidia.com/default/topic/1031300/jetson-tx2/tensorflow-1-8-wheel-with-jetpack-3-2-/) and install using pip. I strongly recomend using TensorFlow 1.10+ , It can download in [here](https://devtalk.nvidia.com/default/topic/1031300/jetson-tx2/tensorflow-1-7-wheel-with-jetpack-3-2-) and install wheel file
   ```
   pip install DOWNLOAD_FILE_NAME.whl
   ```

4. Clone this repository

    ```
    git clone --recursive https://github.com/NVIDIA-Jetson/tf_trt_models.git
    cd tf_trt_models
    ```

5. Run the installation script

    ```
    ./install.sh
    ```
    
    or if you want to specify python intepreter
    
    ```
    ./install.sh python3
    ```

<a name="od"></a>
Object Detection 
----------------

<img src="data/detection_graphic.jpg" alt="detection" height="300px"/>

<a name="od_models"></a>
### Models

| Model | Input Size | TF-TRT TX2 | TF TX2 |
|:------|:----------:|-----------:|-------:|
| ssd_mobilenet_v1_coco | 300x300 | 50.5ms | 72.9ms |
| ssd_inception_v2_coco | 300x300 | 54.4ms | 132ms  |

**TF** - Original TensorFlow graph (FP32)

**TF-TRT** - TensorRT optimized graph (FP16)

The above benchmark timings were gathered after placing the Jetson TX2 in MAX-N
mode.  To do this, run the following commands in a terminal:

```
sudo nvpmodel -m 0
sudo ~/jetson_clocks.sh
```

<a name="od_download"></a>
### Download pretrained model

As a convenience, we provide a script to download pretrained model weights and config files sourced from the
TensorFlow models repository.  

```python
from tf_trt_models.detection import download_detection_model

config_path, checkpoint_path = download_detection_model('ssd_inception_v2_coco')
```
To manually download the pretrained models, follow the links [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

> **Important:** Some of the object detection configuration files have a very low non-maximum suppression score threshold (ie. 1e-8).
> This can cause unnecessarily large CPU post-processing load.  Depending on your application, it may be advisable to raise 
> this value to something larger (like 0.3) for improved performance.  We do this for the above benchmark timings.  This can be done by modifying the configuration
> file directly before calling build_detection_graph.  The parameter can be found for example in this [line](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config#L130).

<a name="od_build"></a>
### Build TensorRT / Jetson compatible graph

```python
from tf_trt_models.detection import build_detection_graph

frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path
)
```

<a name="od_trt"></a>
### Optimize with TensorRT

```python
import tensorflow.contrib.tensorrt as trt

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)
```
### Structure
In **src/optimzie_graph.py** optimzie_graph.py makes trt graph in **dest/frozen**. Have to build using this file, don't use direct .pb  file. Check **src/constants.py** and change the Model and config_prefix own your model format. You can download the model direct from url, Check MODELS in **src/constants.py**

In **src/run_tftrt.py** inference the image, using optimized graph in **dest/frozen/MODEL_NAME_trt.pb** Using images in IMAGE_DIR

<a name="run"></a>
Run
-------------
In root folder
```
python src/run_tftrt.py
```
or if use specific python version
```
python3 src/run_tftrt.py
```

if you don't have optimzie graph, it automatically call optimize_graph in run time
