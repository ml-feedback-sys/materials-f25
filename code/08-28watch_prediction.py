'''
Assignment 08-23: watch prediction
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_DIR = '../../../data/'

def get_kuairec_data(data_dir=DATA_DIR):
	# adapted from https://github.com/chongminggao/KuaiRec/blob/main/loaddata.py
    csv_dir = os.path.join(data_dir, "KuaiRec 2.0", "data")

    try:
        # big_matrix = pd.read_csv(os.path.join(csv_dir,"big_matrix.csv"))
        small_matrix = pd.read_csv(os.path.join(csv_dir,"small_matrix.csv"))
        # social_network = pd.read_csv(os.path.join(csv_dir,"social_network.csv"))
        # social_network["friend_list"] = social_network["friend_list"].map(eval)
        # item_categories = pd.read_csv(os.path.join(csv_dir,"item_categories.csv"))
        # item_categories["feat"] = item_categories["feat"].map(eval)
        user_features = pd.read_csv(os.path.join(csv_dir,"user_features.csv"))
        item_daily_feat = pd.read_csv(os.path.join(csv_dir,"item_daily_features.csv"))
    except FileNotFoundError as e:
        print("Data file not found at", csv_dir)
        print("Have you downloaded the data? See https://kuairec.com/#download-the-data")
    return small_matrix, item_daily_feat, user_features

def extract_features_labels(small_matrix, item_daily_feat, user_features):
	# Joining user and item features
	merged_user_matrix = small_matrix.merge(user_features, how='left', on='user_id')
	merged_matrix = merged_user_matrix.merge(item_daily_feat, how='left', on=['video_id', 'date'])

	# Cleaning and subsampling
	merged_matrix.dropna(inplace=True)
	subsampled_matrix = merged_matrix.sample(n=10000, random_state=0)
	label_name = ['watch_ratio']  # target
	features_name = ['like_cnt', 'comment_cnt', 'music_id', # about video
	                                'follow_user_num_x', 'friend_user_num'] # about user
	data_matrix = subsampled_matrix[label_name + features_name]

	return label_name, features_name, data_matrix                             


# Loading data
small_matrix, item_daily_feat, user_features = get_kuairec_data()
# Merging and extracting features and lavels
label_name, features_name, data_matrix = extract_features_labels(small_matrix, item_daily_feat, user_features)


# Keep the original splitting logic
y_train = np.log(1 + np.array(data_matrix[:9000][label_name]))
X_train = np.array(data_matrix[:9000][features_name])

y_eval = np.log(1 + np.array(data_matrix[9000:][label_name]))
X_eval = np.array(data_matrix[9000:][features_name])

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

# Hyperparameter tuning
best_loss = float('inf')
best_alpha = None
for alpha in [0.01, 0.1, 1, 10]:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    y_pred_eval = model.predict(X_eval_scaled)
    loss = np.mean((y_pred_eval - y_eval)**2)
    print(f'Alpha = {alpha}, Testing loss = {loss}')
    if loss < best_loss:
        best_loss = loss
        best_alpha = alpha

print(f'Best alpha: {best_alpha}')

# Training a Ridge predictor with the best alpha
best_model = Ridge(alpha=best_alpha)
best_model.fit(X_train_scaled, y_train)

# Evaluating the Ridge predictor
y_pred_train = best_model.predict(X_train_scaled)
train_loss = np.mean((y_pred_train - y_train)**2)
print('Training loss:', train_loss)

y_pred_eval = best_model.predict(X_eval_scaled)
eval_loss = np.mean((y_pred_eval - y_eval)**2)
print('Testing loss:', eval_loss)

# Plotting the results
plt.plot(y_eval, y_pred_eval, '.')
plt.xlabel('Actual log-watch ratio')
plt.ylabel('Predicted log-watch ratio')
plt.title('Actual vs. Predicted log-watch ratio')
plt.show()