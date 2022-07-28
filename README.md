# Project Writeup Template
## Project overview: 


The purpose of this project is using convolutional neural network to detect and classify objects using data from Waymo.We Can use camera get more information than other information likes color of detected objects.
## Set up: 
This section should contain a brief description of the steps to follow to run the code for this repository.
Dataset

### 1. using Exploratory Data Analysis.ipynb File to Exploratory Data 
Please open file [Exploratory Data Analysis.ipynb](Exploratory%20Data%20Analysis.ipynb)
### 2. Edit the  pipeline_new.config to improve the result of training
Please open file [pipeline_new.config](pipeline_new.config)
### 3. using Explore augmentations.ipynb File to show the data augmentations result
Please open file [augmentations.ipynb](Explore%20augmentations.ipynb)
### 4. run training code 
``` python
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config

```
### 5. run testsing code
``` python
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/

```
### 6. export model and animation.gif
```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/

python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif

```

Please open file [animation.gif](animation.gif)

## Dataset Analysis: 

We can use the Exploratory Data Analysis file to show the data.

We can figures out the classify case 1 is triple the classify case 2.
**Cross-validation**: we split data to 3 foled: test,train,val.
Train was used in training model,test for check has overfitting.Val data was to evaluating result in real word.

## Training

Now we notice the data set not enough data in low lights situation.So I used adjust_brightness config to light up or down the images.And some Vehichles or Humans will be occlude by other objects So we add crop_image config.
The final result show in image animation.gif.