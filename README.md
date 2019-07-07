# DeepSolar
Nationwide houseshold-level solar panel identification with deep learning. See details from our [project website](http://web.stanford.edu/group/deepsolar/home). We used [Inception-v3](https://arxiv.org/pdf/1512.00567.pdf) as the basic framework for image-level classification and developed greedy layerwise training for segmentation and localization.
CNN model was developed with [TensorFlow](https://github.com/tensorflow). `slim` package is credited to Google. `train_classification.py` and `train_segmentation.py` were developed with reference to [inception](https://github.com/tensorflow/models/tree/master/inception).


## Usage instructions for classification module

Clone repo, pip install requirements.txt and then run

```
find . -name *.jpg | python2 test_classification.py results.csv
```

This will run inference on all jpg images in the directory and print out the probability that they have solar panels installed. There is no need to download the trained model -- this is automatically downloaded from a storage bucket when you run the above script.


### More:

This is a fork of https://github.com/wangzhecheng/DeepSolar, see that repo for more info