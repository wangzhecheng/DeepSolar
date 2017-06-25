""" Generate lists containing filepaths and labels for training, validation and evaluation. """
import pickle
import os.path
import random
import pandas as pd

##### generate training set #####
TRAIN_SET_DIR = 'SPI_train'
train_set_list = []
pos_num = 0
neg_num = 0
# negative samples
for i in xrange(1, 320378):
    img_path = os.path.join(TRAIN_SET_DIR, '0', str(i)+'.png')
    if not os.path.exists(img_path):
        continue
    train_set_list.append((img_path, [0]))
    neg_num += 1

# positive samples
for i in xrange(1, 46091):
    img_path = os.path.join(TRAIN_SET_DIR, '1', str(i)+'.png')
    if not os.path.exists(img_path):
        continue
    train_set_list.append((img_path, [1]))
    pos_num += 1

random.shuffle(train_set_list)

with open('train_set_list.pickle', 'w') as f:
    pickle.dump(train_set_list, f)

print ('Train set list done. # positive samples: '+str(pos_num)+' # negative samples: '+str(neg_num))

##### generate validation set #####
VAL_SET_DIR = 'SPI_val'
val_set_list = []
pos_num = 0
neg_num = 0
# negative samples
for i in xrange(1, 227):
    img_path = os.path.join(VAL_SET_DIR, '0', str(i)+'.png')
    if not os.path.exists(img_path):
        continue
    val_set_list.append((img_path, [0]))
    neg_num += 1

# positive samples
for i in xrange(1, 12761):
    img_path = os.path.join(VAL_SET_DIR, '1', str(i)+'.png')
    if not os.path.exists(img_path):
        continue
    val_set_list.append((img_path, [1]))
    pos_num += 1

with open('val_set_list.pickle', 'w') as f:
    pickle.dump(val_set_list, f)

print ('Validation set list done. # positive samples: '+str(pos_num)+' # negative samples: '+str(neg_num))


##### generate test set #####
TEST_SET_DIR = 'SPI_eval'
test_set_list = []
pos_num = 0
neg_num = 0
eval_set_meta = pd.read_csv(os.path.join(TEST_SET_DIR, 'eval_set_meta.csv')).values
for index in xrange(1, 66):
    region_type = eval_set_meta[index-1, 5] # get the type of the regions
    region_dir = os.path.join(TEST_SET_DIR, str(index))

    # negative samples
    for i in xrange(1, 3001):
        img_path = os.path.join(region_dir, '0', str(i) + '.png')
        if not os.path.exists(img_path):
            continue
        neg_num += 1
        test_set_list.append((img_path, [0], index, i, region_type))

    # positive samples
    for i in xrange(1, 3001):
        img_path = os.path.join(region_dir, '1', str(i) + '.png')
        if not os.path.exists(img_path):
            continue
        pos_num += 1
        test_set_list.append((img_path, [1], index, i, region_type))

with open('test_set_list.pickle', 'w') as f:
    pickle.dump(test_set_list, f)

print ('Test set list done. # positive samples: '+str(pos_num)+' # negative samples: '+str(neg_num))