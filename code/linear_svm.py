import numpy as np
import torch
import torch.nn as nn
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os


def linear_svm(data_path, temporal_process):
  
    folder_list = [folder for folder in sorted(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, folder))]
    print(folder_list)
    
    accuracy = 0

    
    for file in folder_list:  # 每个被试分别处理
        print(f'\nprocessing {file} ...')

        # 获取该被试的数据和标签
        train_data = np.load(f"{data_path}/{file}/train_data.npy")
        train_label = np.load(f"{data_path}/{file}/train_label.npy")
        test_data = np.load(f"{data_path}/{file}/test_data.npy")
        test_label = np.load(f"{data_path}/{file}/test_label.npy")
        
        print(f"train_data: {train_data.shape}, test_data: {test_data.shape}")
        
        # 根据 time_dim_process 参数决定如何处理时间维度
        if temporal_process == 'first':
            train_data = np.squeeze(train_data[:, :, 0, :])
            test_data = np.squeeze(test_data[:, :, 0, :])
        elif temporal_process == 'mean':
            train_data = np.squeeze(np.mean(train_data, axis=2))
            test_data = np.squeeze(np.mean(test_data, axis=2))

        # 将[n, c, v] =》[n, c2]
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)

        print(f"train_data: {train_data.shape}, test_data: {test_data.shape}")

        # 定义svm分类器
        svc_classifier = svm.SVC(C=1, kernel='linear')   # 创建一个SVM分类器实例
        svc_classifier.fit(train_data, train_label)      # 使用训练数据和标签来训练SVM模型
        pred_label = svc_classifier.predict(test_data)   # 使用训练好的模型对测试数据进行预测
        # print(confusion_matrix(test_label, pred_label))
        # print(classification_report(test_label, pred_label))
        cur_accuracy = svc_classifier.score(test_data, test_label)
        accuracy += cur_accuracy
        print(f'accuracy: {cur_accuracy}')

    print('\nall experiment average accuracy: {}'.format(accuracy / len(folder_list)))

if __name__ == "__main__": 
    data_path = '/home/wuchennan/Dataset/seed3/sub_independent'
    linear_svm(data_path, temporal_process="first")
    