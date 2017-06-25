import pickle
import numpy as np
import os.path
import skimage.io
import matplotlib.pyplot as plt
import seaborn
from sklearn import linear_model
import scipy.stats

PRED_DIR = '/Users/wzc/Documents/DeepSolar/segmentation_result/eval_seg_results/TP/'
PRED_DIR_FP = '/Users/wzc/Documents/DeepSolar/segmentation_result/eval_seg_results/FP/'
"""GROUND_TRUTH_DIR = '/Users/wzc/Documents/DeepSolar/segmentation/seg_eval_set/TP/'
# generate ground truth areas
area_dict = {}
counter =0
for i in xrange(1, 1088):
    ground_truth_path = GROUND_TRUTH_DIR+str(i)+'_mask.png'
    if not os.path.exists(ground_truth_path):
        continue
    img = skimage.io.imread(ground_truth_path)
    img = img/255.0
    area = sum(sum(img))
    area = area * (100*100)/(320*320)
    print area
    area_dict[i] = area
    counter+=1

with open('TP_ground_truth_area.pickle', 'w') as f:
    pickle.dump(area_dict, f)

print counter"""
downtown_index_list = [7, 8, 9, 14, 15, 35, 38, 39, 42, 19, 20, 22, 23, 26, 29, 30, 31, 59, 68, 70, 46, 67, 51, 52, 54, 61, 63]
residential_index_list = [10, 11, 12, 13, 16, 17, 34, 36, 37, 40, 41, 18, 21, 24, 25, 27, 28, 32, 33, 57, 58, 66, 67, 69, 71, 43, 44, 45, 48, 49, 50, 53, 55, 56, 60, 62, 64, 65]
DOWNTOWN_INDEX_LIST = [d-1 for d in downtown_index_list]
RESIDENTIAL_INDEX_LIST = [d-1 for d in residential_index_list]


with open('TP_ground_truth_area.pickle', 'r') as f:
    area_dict = pickle.load(f)

with open('FN_ground_truth_area.pickle', 'r') as f:
    area_dict_FN = pickle.load(f)

with open('/Users/wzc/Documents/DeepSolar/big_data/TP_seg_info.pickle', 'r') as f:
    info = pickle.load(f)

with open('/Users/wzc/Documents/DeepSolar/big_data/FP_seg_info.pickle', 'r') as f:
    info_FP = pickle.load(f)

with open('/Users/wzc/Documents/DeepSolar/big_data/FN_seg_info.pickle', 'r') as f:
    info_FN = pickle.load(f)


error_rate_list = []
threshold_list = [0.37]
best_threshold = 0
min_error = 1000000
pred = []
true = []
error = []
error2 = []
for THRES in threshold_list:
    #print THRES
    diff = 0
    sum = 0
    for i in xrange(1, 1088):
        pred_path = PRED_DIR+str(i)+'_CAM.png'
        if not os.path.exists(pred_path):
            continue
        if not info[i][0] in RESIDENTIAL_INDEX_LIST:
            continue
        heat_map = skimage.io.imread(pred_path)
        #heat_map = np.load(pred_path)
        heat_map = heat_map/65535.0

        pred_area = np.sum(heat_map>THRES)

        area_diff = np.abs(pred_area - area_dict[i])
        diff += area_diff
        sum += area_dict[i]
        error.append((pred_area - area_dict[i])*0.18*0.18)


        pred.append(pred_area*0.18*0.18)
        true.append([area_dict[i]*0.18*0.18])
    # FP
    """for i in xrange(1, 140):
        pred_path = PRED_DIR_FP + str(i) + '_CAM.png'
        if not os.path.exists(pred_path):
            continue
        if not info_FP[i][0] in RESIDENTIAL_INDEX_LIST:
            continue
        heat_map = skimage.io.imread(pred_path)
        heat_map = heat_map / 65535.0
        pred_area = np.sum(heat_map > THRES)
        error.append(pred_area * 0.18 * 0.18)
# FN
for i in xrange(1, 136):
    if not info_FN[i][0] in RESIDENTIAL_INDEX_LIST:
        continue
    error.append(-area_dict_FN[i] * 0.18 * 0.18)"""

    """error_rate = float(diff)/sum
    if error_rate < min_error:
        min_error = error_rate
        best_threshold = THRES
    error_rate_list.append(error_rate)"""

#print best_threshold
#print min_error

"""regr = linear_model.LinearRegression()
regr.fit(true, pred)
intercept = regr.intercept_
coef = regr.coef_[0]

plt.scatter(true, pred, color='blue')
a, = plt.plot(true, regr.predict(true), color='red')
plt.xlabel('ground truth area (square meters)')
plt.ylabel('estimated area (square meters)')
plt.legend([a], ["coef: "+str(coef)+" intercept: "+str(intercept)])
plt.title("Correlation between ground truth area and estimated area")
#a, = plt.plot(threshold_list, error_rate_list)
plt.show()"""

#print len(error)

#print scipy.stats.kstest(error, "norm")

for THRES in threshold_list:
    #print THRES
    diff = 0
    sum = 0
    for i in xrange(1, 1088):
        pred_path = PRED_DIR+str(i)+'_CAM.png'
        if not os.path.exists(pred_path):
            continue
        if not info[i][0] in DOWNTOWN_INDEX_LIST:
            continue
        heat_map = skimage.io.imread(pred_path)
        #heat_map = np.load(pred_path)
        heat_map = heat_map/65535.0

        pred_area = np.sum(heat_map>THRES)

        area_diff = np.abs(pred_area - area_dict[i])
        diff += area_diff
        sum += area_dict[i]
        error2.append((pred_area - area_dict[i])*0.18*0.18)


        pred.append(pred_area*0.18*0.18)
        true.append([area_dict[i]*0.18*0.18])
    # FP
    """for i in xrange(1, 140):
        pred_path = PRED_DIR_FP + str(i) + '_CAM.png'
        if not os.path.exists(pred_path):
            continue
        if not info_FP[i][0] in DOWNTOWN_INDEX_LIST:
            continue
        heat_map = skimage.io.imread(pred_path)
        heat_map = heat_map / 65535.0
        pred_area = np.sum(heat_map > THRES)
        error2.append(pred_area * 0.18 * 0.18)
# FN
for i in xrange(1, 136):
    if not info_FN[i][0] in DOWNTOWN_INDEX_LIST:
        continue
    error2.append(-area_dict_FN[i] * 0.18 * 0.18)"""




print len(error)
print len(error2)

error_array = np.array(error)
error_array_2 = np.array(error2)

print("Residential mean: "+str(np.mean(error))+"std: "+str(np.std(error)))
print("Commerical mean: "+str(np.mean(error2))+"std: "+str(np.std(error2)))

plt.subplot(1, 2, 1)
plt.hist(error, bins=80, color='blue')
plt.xlim([-200, 200])
plt.title('Distribution of Area Difference (m^2) - Residential')

plt.subplot(1, 2, 2)
plt.hist(error, bins=50, color='blue')
plt.xlim([-200, 200])
plt.title('Distribution of Area Difference (m^2) - Commercial')

plt.show()