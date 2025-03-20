import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from linear_svm import linear_svm
import scipy.io as sio
import pickle
import sys

# 获取数据集的基本参数信息
def get_dataset_info(dataset):
    if dataset == "seed3":
        fs = 200
        num_subjects = 15
        num_trials = 15
        segment_len = 1  # 默认SEED3使用1秒的分段长度
        labels = [
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
            [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
        ]
    elif dataset == "seed4":
        fs = 200
        num_trials = 24
        num_subjects = 15
        segment_len = 4  # 默认SEED4使用4秒的分段长度
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
    
    return {
        'fs': fs,
        'num_subjects': num_subjects,
        'num_trials': num_trials,
        'labels': labels,
        'segment_len': segment_len
    }

# 将EEG数据分割为segment_len秒片段并生成对应标签
def segment_eeg_data(trial_data, trial_label, fs, segment_len):
    """
    将EEG数据分割为segment_len秒的片段并生成对应标签
    
    参数:
    - trial_data: 一个trial的EEG数据, 形状为[C, T]的
    - trial_label: 当前trial的标签(一个数字)
    - fs: 采样频率
    
    返回:
    - segmented_data: 分割后的数据，形状为 [N, C, fs]
    - segmented_label: 对应的标签，形状为 [N]
    """
    
    # 分割数据成指定长度片段
    C, T = trial_data.shape
    points_per_segment = segment_len * fs
    num_segments = T // points_per_segment                           # 计算可以分割出多少个完整的片段
    segments = []
    for seg_idx in range(num_segments):
        start_idx = seg_idx * points_per_segment
        end_idx = (seg_idx + 1) * points_per_segment
        segment = trial_data[:, start_idx:end_idx]
        segments.append(segment)
    
    segmented_data = np.stack(segments, axis=0) if segments else np.array([])            # 将所有片段堆叠，形状为 [N, C, points_per_segment]
    segmented_label = np.full(len(segments), trial_label) if segments else np.array([])  # 为每个分割片段生成相同的标签

    return segmented_data, segmented_label
    
# 处理单个被试的数据并分割为训练集和测试集
def segment_per_subject(data_path, dataset, subject_id):
    """
    处理单个被试的数据并返回训练集和测试集
    
    参数:
    - data_path: 数据路径
    - dataset: 数据集名称
    - subject_id: 被试ID
    
    返回:
    - train_data: 该被试的训练数据 (前16个trial)
    - train_labels: 该被试的训练标签
    - test_data: 该被试的测试数据 (后8个trial)
    - test_labels: 该被试的测试标签
    """
    # 获取数据集对应的信息
    dataset_info = get_dataset_info(dataset)
    
    # 存储该被试的训练集和测试集
    train_data_sessions = []
    train_labels_sessions = []
    test_data_sessions = []
    test_labels_sessions = []
    
    # 依次处理该被试的每个session
    for session in os.listdir(data_path):
        session_id = int(session[-1])                        # 获取session的ID
        session_dir = os.path.join(data_path, session)  

        # 查找当前被试在该session中的文件并加载
        subject_file = None
        for file in os.listdir(session_dir):
            if file.startswith(f"{subject_id}_") and file.endswith(".mat"):
                subject_file = file
                break
        if subject_file is None:
            raise FileNotFoundError(f"在session{session_id}中找不到被试{subject_id}的文件")
        file_path = os.path.join(session_dir, subject_file)
        print(f"-处理session{session_id}, 文件: {subject_file}")
        mat_data = sio.loadmat(file_path)

        # 分别处理训练集(前16个trial)和测试集(后8个trial)
        train_trials_data = []
        train_trials_labels = []
        test_trials_data = []
        test_trials_labels = []
        
        for trial_idx in range(1, dataset_info['num_trials']+1):
            # 获取当前trial在该文件中对应的变量
            matching_var = None
            for key in mat_data.keys():
                # 检查变量名是否是字符串类型
                if isinstance(key, str) and key.endswith(str(trial_idx)):
                    matching_var = key
                    break
            if matching_var:
                trial_data = mat_data[key]                                                          # 获取当前trial的数据
                dataset_labels = dataset_info['labels']                                              
                trial_label = dataset_labels[session_id-1][trial_idx-1]                             # 获取当前trial的标签

                segemnt_trial_data, segemnt_trial_labels = segment_eeg_data(
                    trial_data, trial_label, dataset_info['fs'], dataset_info['segment_len'])       # 分割数据
                
                print(f"--trial{trial_idx}: data.shape: {trial_data.shape}, label: {trial_label} ==> "
                      f"分割后: {segemnt_trial_data.shape[0]}个片段，数据形状: {segemnt_trial_data.shape}，"
                      f"标签形状: {segemnt_trial_labels.shape}")

                # 前16个trial作为训练集，后8个trial作为测试集
                if trial_idx <= 16:
                    train_trials_data.append(segemnt_trial_data)
                    train_trials_labels.append(segemnt_trial_labels)
                else:
                    test_trials_data.append(segemnt_trial_data)
                    test_trials_labels.append(segemnt_trial_labels)

        # 合并当前session的数据
        if train_trials_data:
            session_train_data = np.concatenate(train_trials_data, axis=0)
            session_train_labels = np.concatenate(train_trials_labels, axis=0)
            train_data_sessions.append(session_train_data)
            train_labels_sessions.append(session_train_labels)
            print(f"--会话{session_id}训练集数据形状: {session_train_data.shape}，标签形状: {session_train_labels.shape}")
        
        if test_trials_data:
            session_test_data = np.concatenate(test_trials_data, axis=0)
            session_test_labels = np.concatenate(test_trials_labels, axis=0)
            test_data_sessions.append(session_test_data)
            test_labels_sessions.append(session_test_labels)
            print(f"--会话{session_id}测试集数据形状: {session_test_data.shape}，标签形状: {session_test_labels.shape}")

    # 合并所有session的数据
    if train_data_sessions:
        subject_train_data = np.concatenate(train_data_sessions, axis=0)
        subject_train_labels = np.concatenate(train_labels_sessions, axis=0)
        print(f"被试{subject_id}训练集总数据形状: {subject_train_data.shape}，标签形状: {subject_train_labels.shape}")
    else:
        subject_train_data = np.array([])
        subject_train_labels = np.array([])
    
    if test_data_sessions:
        subject_test_data = np.concatenate(test_data_sessions, axis=0)
        subject_test_labels = np.concatenate(test_labels_sessions, axis=0)
        print(f"被试{subject_id}测试集总数据形状: {subject_test_data.shape}，标签形状: {subject_test_labels.shape}")
    else:
        subject_test_data = np.array([])
        subject_test_labels = np.array([])
    
    return subject_train_data, subject_train_labels, subject_test_data, subject_test_labels

def process_all_subjects(data_path, save_path, dataset):
    """
    处理所有被试的EEG数据，生成一个合并的训练集和测试集
    
    参数:
    - data_path: 数据的路径
    - save_path: 保存结果的路径
    - dataset: 数据集名称
    """
    dataset_info = get_dataset_info(dataset)
    subject_ids = list(range(1, dataset_info['num_subjects'] + 1))
    print(f"待处理的被试ID: {subject_ids}")
    
    # 存储所有被试的训练集和测试集数据
    all_train_data = []
    all_train_labels = []
    all_test_data = []
    all_test_labels = []
    
    for subject_id in subject_ids:
        print(f"\n开始处理被试{subject_id}")
        try:
            # 获取单个被试的训练集和测试集
            subject_train_data, subject_train_labels, subject_test_data, subject_test_labels = segment_per_subject(
                data_path, dataset, subject_id
            )
            
            # 将该被试的数据添加到总数据集中
            if len(subject_train_data) > 0:
                all_train_data.append(subject_train_data)
                all_train_labels.append(subject_train_labels)
            
            if len(subject_test_data) > 0:
                all_test_data.append(subject_test_data)
                all_test_labels.append(subject_test_labels)
                
        except Exception as e:
            print(f"处理被试{subject_id}时出错: {e}")
    
    # 合并所有被试的数据
    if all_train_data:
        combined_train_data = np.concatenate(all_train_data, axis=0)
        combined_train_labels = np.concatenate(all_train_labels, axis=0)
        print(f"\n所有被试合并后的训练集数据形状: {combined_train_data.shape}，标签形状: {combined_train_labels.shape}")
    else:
        print("没有可用的训练数据")
        return
    
    if all_test_data:
        combined_test_data = np.concatenate(all_test_data, axis=0)
        combined_test_labels = np.concatenate(all_test_labels, axis=0)
        print(f"\n所有被试合并后的测试集数据形状: {combined_test_data.shape}，标签形状: {combined_test_labels.shape}")
    else:
        print("没有可用的测试数据")
        return
    
    # 创建训练集和测试集字典
    train_dict = {
        'data': combined_train_data,
        'label': combined_train_labels
    }
    
    test_dict = {
        'data': combined_test_data,
        'label': combined_test_labels
    }
    
    # 保存训练集和测试集到不同文件
    os.makedirs(save_path, exist_ok=True)
    
    train_file_path = os.path.join(save_path, f"{dataset}_train.pkl")
    with open(train_file_path, 'wb') as file:
        pickle.dump(train_dict, file)
    print(f"\n训练集已保存到: {train_file_path}")
    
    test_file_path = os.path.join(save_path, f"{dataset}_test.pkl")
    with open(test_file_path, 'wb') as file:
        pickle.dump(test_dict, file)
    print(f"\n测试集已保存到: {test_file_path}")
    
    return train_dict, test_dict

if __name__ == "__main__":
    root_path = "/home/wuchennan/Dataset"
    dataset = ["seed3", "seed4", "seed5"][1]            # 选择数据集
    data_type = ["eeg_preprocessed", "eeg_feature"][0]     # 选择数据类型
    
    data_path = os.path.join(root_path, "raw_data", dataset, data_type)
    save_path = os.path.join(root_path, "segment_data", dataset, data_type)
    
    # 处理所有被试的数据并生成合并的训练集和测试集
    combined_dataset = process_all_subjects(data_path, save_path, dataset)