import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from linear_svm import linear_svm

root_path = "/home/wuchennan/Dataset"
freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]

# 对输入数据进行划窗处理
def sliding_window(data, window_length, step):
    """ 对输入数据进行划窗处理。
    输入: data : (channels, time, features)
    返回: windowed_data : 划窗后的数据，形状为 (num_windows, channels, window_length, features)
    """
    # 计算可以形成多少个窗口
    num_windows = (data.shape[1] - window_length) // step + 1
    if num_windows <= 0:
        raise ValueError("Window length and step size are too large for the provided data length.")
    
    # 初始化窗口数据数组,[N, C, T, V]
    windowed_data = np.zeros((num_windows, data.shape[2], window_length, data.shape[0]))  

    # 创建窗口
    for j in range(num_windows):
        start_idx = j * step
        end_idx = start_idx + window_length
        windowed_data[j] = data[:, start_idx:end_idx, :].transpose(2, 1, 0)  # [v, t, c] -> [c, t, v]

    return windowed_data

# 划分
def segment_signal(signal, fs=200, window_size=1):
    """
    将形状为[C, T]的信号数据分割成形状为[N, C, window_points]的数据段
    
    参数:
    signal: 形状为[C, T]的numpy数组，C是通道数，T是总采样点数
    fs: 采样频率，默认为200 Hz
    window_size: 每个窗口的时间长度（秒），默认为1秒
    
    返回:
    segments: 形状为[N, C, window_points]的numpy数组，N是段数，window_points是每段的采样点数
    """
    C, T = signal.shape
    window_points = int(fs * window_size)  # 每个窗口的采样点数，这里是200
    
    # 计算可以分成多少个完整的段
    n_segments = T // window_points
    
    # 如果总采样点数不能被窗口采样点数整除，会丢弃尾部的采样点
    if n_segments == 0:
        raise ValueError(f"信号长度 ({T}) 小于一个窗口长度 ({window_points})，无法分段")
    
    # 初始化结果数组
    segments = np.zeros((n_segments, C, window_points))
    
    # 分段
    for i in range(n_segments):
        start = i * window_points
        end = start + window_points
        segments[i] = signal[:, start:end]
    
    return segments

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

def build_de_libarary(dataset, window_length, step):
    # 获取num_trials和labels
    num_trials, labels = get_variables(dataset)
    
    # 构建实际的数据路径
    data_path = f"{root_path}/{dataset}/extracted_de"

    # 初始化两个字典用于存储每个被试的数据和标签
    data_dict, label_dict = {}, {}
    
    for session_id in os.listdir(data_path):
        session_path = os.path.join(data_path, session_id)
        file_list = [file for file in os.listdir(session_path)]
        for file in tqdm(file_list, desc=f"Building DE Libarary for session{session_id}"):
            subject_id = file.split('_')[0]                        # 获取文件名
            data = scio.loadmat(os.path.join(session_path, file))  # 获取每个文件对应的数据

            data_trial_dict = {}                                # 存放每个trial数据的字典, key: trial, value: data
            label_trial_dict = {}                               # 存放每个trial标签的字典, key: trial, value: label

            for trial in range(1, num_trials+1):
                trial_data = data['de_LDS' + str(trial)]                                # [v, t, c]
                one_trial_data = sliding_window(trial_data, window_length, step)        # (34, 5, 9, 62)
                one_trial_labels = np.full(one_trial_data.shape[0], labels[int(session_id)-1][trial - 1])  
                
                # 存储为对应的键
                data_trial_dict[str(trial)] = one_trial_data
                label_trial_dict[str(trial)] = one_trial_labels

            data_dict[f"sub{int(subject_id):02d}_session{session_id}"] = data_trial_dict
            label_dict[f"sub{int(subject_id):02d}_session{session_id}"] = label_trial_dict

    return data_dict, label_dict

def build_raw_libarary(dataset, window_length, step):
    # 获取num_trials和labels
    num_trials, labels = get_variables(dataset)
    
    # 构建实际的数据路径
    data_path = f"{root_path}/{dataset}/eeg_raw_data"

    # 初始化两个字典用于存储每个被试的数据和标签
    data_dict, label_dict = {}, {}
    
    for session_id in os.listdir(data_path):
        session_path = os.path.join(data_path, session_id)
        file_list = [file for file in os.listdir(session_path)]
        for file in tqdm(file_list, desc=f"Building Raw EEG Libarary for session{session_id}"):
            subject_id = file.split('_')[0]                        # 获取文件名
            data = scio.loadmat(os.path.join(session_path, file))  # 获取每个文件对应的数据

            data_trial_dict = {}                                # 存放每个trial数据的字典, key: trial, value: data
            label_trial_dict = {}                               # 存放每个trial标签的字典, key: trial, value: label

            for trial in range(1, num_trials+1):
                trial_data = data['cz_EEG' + str(trial)]                                # [v, t, c]
                one_trial_data = segment_signal(trial_data)        # (34, 5, 9, 62)
                one_trial_labels = np.full(one_trial_data.shape[0], labels[int(session_id)-1][trial - 1])  
                
                # 存储为对应的键
                data_trial_dict[str(trial)] = one_trial_data
                label_trial_dict[str(trial)] = one_trial_labels

            data_dict[f"sub{int(subject_id):02d}_session{session_id}"] = data_trial_dict
            label_dict[f"sub{int(subject_id):02d}_session{session_id}"] = label_trial_dict

    return data_dict, label_dict

def gen_dependent_data(data_path, save_path, window_length, step):

    data_dict, label_dict = build_de_libarary(data_path, window_length, step)  # 获取DE特征数据集

    for file, file_data in data_dict.items():
        train_data, train_label = [], []
        test_data, test_label = [], []

        # 遍历该文件中的所有trial
        for trial, trial_data in file_data.items():
            labels = label_dict[file][trial]
            if int(trial) <= 9:
                train_data.append(trial_data)
                train_label.extend(labels)
            else:
                test_data.append(trial_data)
                test_label.extend(labels)

        train_data = np.concatenate(train_data, axis=0) if train_data else np.array([])
        test_data = np.concatenate(test_data, axis=0) if test_data else np.array([])
        train_label = np.array(train_label)
        test_label = np.array(test_label)

        # 判断标签中是否存在负数, 如果存在则将所有元素加1
        if np.any(train_label < 0) or np.any(test_label < 0):
            train_label += 1
            test_label += 1

        print(f'{file}: train_data: {train_data.shape}, test_data: {test_data.shape}')

        # 在save_path目录下创建以file命名的文件夹
        file_path = os.path.join(save_path, file)
        os.makedirs(file_path, exist_ok=True)

        # 将数据保存为npy文件
        np.save(os.path.join(file_path, 'train_data.npy'), train_data)
        np.save(os.path.join(file_path, 'train_label.npy'), train_label)
        np.save(os.path.join(file_path, 'test_data.npy'), test_data)
        np.save(os.path.join(file_path, 'test_label.npy'), test_label)

def gen_independent_data(dataset, window_length, step):
    # data_dict, label_dict = build_de_libarary(dataset, window_length, step)  # 获取DE特征数据集
    data_dict, label_dict = build_raw_libarary(dataset, window_length, step)

    # 将文件按被试分类
    subject_files = {}
    for file in data_dict.keys():
        subject_id = file.split("_")[0]
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(file)
    print(f"subject_files: {subject_files}")

    for test_subject, _ in subject_files.items():
        train_data, train_label = [], []
        test_data, test_label = [], []

        for file, file_data in data_dict.items():
            subject_id = file.split("_")[0]

            # 如果当前文件属于测试被试，将其数据作为测试集
            if subject_id == test_subject:
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
    step=200
    window_length=200
    dataset = ["seed3", "seed4", "seed5"][1]            # 选择数据集
    protocol = ["sub_dependent", "sub_independent"][1]  # 选择试验协议

    # 生成数据集
    data_path = f"{root_path}/{dataset}/1s_raw_data"
    save_path = f"{root_path}/{dataset}/{protocol}"
    if protocol == "sub_dependent":
        gen_dependent_data(dataset, window_length, step)
    else:
        gen_independent_data(dataset, window_length, step)

    # 对保存的数据进行校验
    linear_svm(save_path, temporal_process="first")  
