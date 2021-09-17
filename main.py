from mine_function import *
import sys
import tensorflow as tf
sys.path.append("..")
from DeepFM import DeepFM


#A = read_txt1("./data/lncRNADisease2.txt")
A = read_txt1("./data/miRNA database/HMDD3.2.txt")
print("the number of biomarkers and diseases", A.shape)
print("the number of known associations", sum(sum(A)))
x,y = A.shape
#get all the training samples and all the unknown negative samples
samples, neg = get_balance_samples(A)
label_all = []
y_score_all = []
print("sample_size", samples.shape)

# ------------------ DeepFM Model ------------------
# params
dfm_params = {
    "embedding_size": 3,
    "dropout_fm": [1, 1],
    "deep_layers": [64, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch":30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 0,
    "batch_norm_decay": 0.995,
    "verbose": True,
    "l2_reg": 0.001,
}

dfm_params["feature_size"] = 2*(x+y)
dfm_params["field_size"] = x+y
roc_sum = 0

k =1
pre_total =[]
label_total = []

train_fea, y_train_ = get_feature_label(A, samples, k)
Xi_train_, Xv_train_ = data_transform(train_fea, k)

#train the model
dfm = DeepFM(**dfm_params)
dfm.fit(Xi_train_, Xv_train_, y_train_)

# begin to predict, as there are too much associations need to be predicted, we divide them to saveral tasks, each task predicts 10000 associations
new_ass_matrix = np.zeros([x, y])
test_len = neg.shape[0]
temp_it = int(test_len / 10000) + 1
for i in range(temp_it):
    if i < (temp_it - 1):
        Xi_test_, Xv_test_ = data_transform2(A, neg[(i * 10000): (i + 1) * 10000, :], k)
        pre = []
        pre = dfm.predict(Xi_test_, Xv_test_)
        for j in range(10000):
            new_ass_matrix[neg[i * 10000 + j, 0], neg[i * 10000 + j,]] = pre[j]
    else:
        Xi_test_, Xv_test_ = data_transform2(A, neg[(i * 10000): test_len, :], k)
        pre = []
        pre = dfm.predict(Xi_test_, Xv_test_)
        for j in range(test_len - (i * 10000)):
            new_ass_matrix[neg[i * 10000 + j, 0], neg[i * 10000 + j, 1]] = pre[j]

m2, n2 = samples.shape
for i in range(m2):
    new_ass_matrix[samples[i, 0], samples[i, 1]] = 1

#Save the prediction result
np.savetxt('predicted matrix.txt', new_ass_matrix)


