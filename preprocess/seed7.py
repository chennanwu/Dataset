import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from linear_svm import linear_svm
import mne

root_path = "/home/wuchennan/Dataset"
freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]
fs = 200

def sliding_window(data, window_length, step):
    
    """ 对输入数据进行划窗处理。
    输入: data : (channels, time, features)
    返回: windowed_data : 划窗后的数据，形状为 (num_windows, channels, window_length, features)
    """
    if data.shape[1] < window_length:
        raise ValueError("Data length is shorter than window length.")
    
    # 计算完整窗口的数量（向下取整，丢弃不足一个窗口的部分）
    num_windows = (data.shape[1] - window_length) // step + 1
    
    # 检查是否有足够的数据形成至少一个窗口
    if num_windows <= 0:
        raise ValueError("Window length and step size are too large for the provided data length.")
    
    # 初始化窗口数据数组 [N, C, T, V]
    windowed_data = np.zeros((num_windows, data.shape[2], window_length, data.shape[0]))
    
    # 创建窗口，只处理能形成完整窗口的数据
    for j in range(num_windows):
        start_idx = j * step
        end_idx = start_idx + window_length
        # 确保不超出数据范围
        if end_idx <= data.shape[1]:
            windowed_data[j] = data[:, start_idx:end_idx, :].transpose(2, 1, 0)  # [v, t, c] -> [c, t, v]
    
    return windowed_data


def get_variables(dataset):
    if dataset == "seed3":
        num_trials = 15
        labels = [
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
        ]
    elif dataset == "seed4":
        num_trials = 24
        labels = [
            [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
            [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
            [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
        ]
    elif dataset == "seed7":
        num_trials = 80
        labels = [0, 6, 2, 1, 5, 5, 1, 2, 6, 0, 0, 6, 2, 1, 5, 5, 1, 2, 6, 0, 5, 1, 3, 6, 4, 4, 6, 3, 1, 5, 5, 1, 3, 6, 4, 4, 6, 3, 1, 5, 0, 4, 2, 3, 5, 5, 3, 2, 4, 0, 0, 4, 2, 3, 5, 5, 3, 2, 4, 0, 2, 1, 3, 4, 0, 0, 4, 3, 1, 2, 2, 1, 3, 4, 0, 0, 4, 3, 1, 2]
    else:
        raise ValueError(f"未找到数据集：{dataset}")
    
    return num_trials, labels


def build_de_library(dataset, window_length, step):
    # 获取num_trials和labels
    num_trials, labels = get_variables(dataset)
    
    # 构建实际的数据路径
    data_path = f"{root_path}/{dataset}/eeg_preprocessed"

    # 初始化两个字典用于存储每个被试的数据和标签
    data_dict, label_dict = {}, {}
    
    file_list = [file for file in os.listdir(data_path)]
    for file in file_list:
        subject_id = file.split('.')[0]                        # 获取文件名
        data = scio.loadmat(os.path.join(data_path, file))     # 获取每个文件对应的数据

        data_trial_dict = {}                                   # 存放每个trial数据的字典, key: trial, value: data
        label_trial_dict = {}                                  # 存放每个trial标签的字典, key: trial, value: label

        for trial in tqdm(range(1, num_trials+1), desc=f"Building DE Libarary for file {file}"):
            trial_data = data[str(trial)]                      # [通道, 12400]
            
            # 每秒提de特征
            de_five = []
            for freq in range(len(freq_bands)):
                low_freq = freq_bands[freq][0]                 # 获取这个频率的起始和结束HZ
                high_freq = freq_bands[freq][1]
                trial_data_filt = mne.filter.filter_data(trial_data, fs, l_freq=low_freq, h_freq=high_freq, verbose=False)  # 获取这个频带的数据
                trial_data_filt = trial_data_filt.reshape(trial_data.shape[0], -1, fs) # [通道，1s窗口的个数，fs]
                de_one = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(trial_data_filt, 2)))  # 提取该频段的de特征
                de_five.append(de_one)            # 将当前频带的DE特征添加到列表中
            de_five = np.stack(de_five, axis=-1)  # shape: [通道数, 秒数, 频带数]

            one_trial_data = sliding_window(de_five, window_length, step)             # (num_windows, channels, window_length, features)
            one_trial_labels = np.full(one_trial_data.shape[0], labels[int(trial)-1]) # (标签数)
            
            # 存储为对应的键
            data_trial_dict[str(trial)] = one_trial_data
            label_trial_dict[str(trial)] = one_trial_labels

        data_dict[f"sub{int(subject_id):02d}"] = data_trial_dict
        label_dict[f"sub{int(subject_id):02d}"] = label_trial_dict

    return data_dict, label_dict

def gen_independent_data(dataset, window_length, step):
    data_dict, label_dict = build_de_library(dataset, window_length, step)  # 获取DE特征数据集

    for test_subject, _ in data_dict.items():
        train_data, train_label = [], []
        test_data, test_label = [], []

        for file, file_data in data_dict.items():
            # 如果当前文件属于测试被试，将其数据作为测试集
            if file == test_subject:
                for trial, trial_data in file_data.items():
                    test_data.append(trial_data)
                    test_label.extend(label_dict[file][trial])

            # 否则，将其数据作为训练集
            else:
                for trial, trial_data in file_data.items():
                    train_data.append(trial_data)
                    train_label.extend(label_dict[file][trial])

        train_data = np.concatenate(train_data, axis=0) if train_data else np.array([])
        test_data = np.concatenate(test_data, axis=0) if test_data else np.array([])
        train_label = np.array(train_label)
        test_label = np.array(test_label)

        # 判断标签中是否存在负数, 如果存在则将所有元素加1
        if np.any(train_label < 0) or np.any(test_label < 0):
            train_label += 1
            test_label += 1

        print(f'Test subject: {test_subject}, train_data: {train_data.shape}, test_data: {test_data.shape}')

        # 在save_path目录下创建以被试id命名的文件夹
        subject_path = os.path.join(save_path, test_subject)
        os.makedirs(subject_path, exist_ok=True)

        # 将数据保存为npy文件
        np.save(os.path.join(subject_path, 'train_data.npy'), train_data)
        np.save(os.path.join(subject_path, 'train_label.npy'), train_label)
        np.save(os.path.join(subject_path, 'test_data.npy'), test_data)
        np.save(os.path.join(subject_path, 'test_label.npy'), test_label)

if __name__ == "__main__": 
    
    # 设置数据集的信息
    step = 4  # 单位秒
    window_length = 4
    dataset = ["seed3", "seed4", "seed7"][2]            # 选择数据集
    protocol = ["sub_dependent", "sub_independent"][1]  # 选择试验协议

    # 生成数据集
    data_path = f"{root_path}/{dataset}/extracted_de"
    save_path = f"{root_path}/{dataset}/{protocol}"
    if protocol == "sub_dependent":
        gen_dependent_data(dataset, window_length, step)
    else:
        gen_independent_data(dataset, window_length, step)

    # 对保存的数据进行校验
    linear_svm(save_path, temporal_process="first")  