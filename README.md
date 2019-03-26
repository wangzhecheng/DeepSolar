# DeepSolar
Nationwide houseshold-level solar panel identification with deep learning. See details from our [project website](http://web.stanford.edu/group/deepsolar/home). We used [Inception-v3](https://arxiv.org/pdf/1512.00567.pdf) as the basic framework for image-level classification and developed greedy layerwise training for segmentation and localization.
CNN model was developed with [TensorFlow](https://github.com/tensorflow). `slim` package is credited to Google. `train_classification.py` and `train_segmentation.py` were developed with reference to [inception](https://github.com/tensorflow/models/tree/master/research/inception/inception). The inception library should be downloaded from [this source](https://github.com/tensorflow/models/tree/master/research/inception/inception). The model was developed with Python 2.7.

### Usage Instructions:
```
git clone https://github.com/wangzhecheng/DeepSolar.git
cd DeepSolar
```
The model is fine-tuned based on the pre-trained model. The pre-trained model was trained on ImageNet 2012 Challenge training set. It can be downloaded as follows:
```
mkdir ckpt
cd ckpt
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
tar xzf inception-v3-2016-03-01.tar.gz
```
Then download pre-trained classification model and segmentation model for solar panel identification task.
```
curl -O https://s3-us-west-1.amazonaws.com/roofsolar/inception_classification.tar.gz
tar xzf inception_classification.tar.gz
curl -O https://s3-us-west-1.amazonaws.com/roofsolar/inception_segmentation.tar.gz
tar xzf inception_segmentation.tar.gz
```
Because the restriction of data sources, we are sorry that we cannot make the training and test set publicly available currently.

Install the required packages:
```
pip install -r requirements.txt
```
Firstly, you should generate data file path lists for training and evaluation. Here is the example:
```
python generate_data_list.py
```
Then you can train the CNN model for classification. You can start from ImageNet model:
```
python train_classification.py --fine_tune=False
```
or start from our well-trained model:
```
python train_classification.py --fine_tune=True
```
After training is done, test the model:
```
python test_classification.py
```
Our model can achieved overall recall 88.9% and overall precision 93.2% on test set.
For training the segmentation branch, you should firstly train the first layer:
```
python train_segmentation.py --two_layers=False
```
Then train the second layer.
```
python train_segmentation.py --two_layers=True
```
After training is done, you can test the average absolute area error rate:
```
python test_segmentation.py
```
Our well-trained model can reach 27.3% for residential area and 18.8% for commercial area.
