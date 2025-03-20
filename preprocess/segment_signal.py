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

# 将EEG数据分割为1秒片段并生成对应标签
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
    
# 处理单个被试的所有session数据
def segment_per_subject(data_path, save_path, dataset, subject_id):
    # 获取数据集对应的信息
    dataset_info = get_dataset_info(dataset)
    
    # 依次处理该被试的每个session
    one_subject_all_sessions_data = []    # 存储一个被试所有session的数据和标签，每个session是列表中的一个元素
    one_subject_all_sessions_labels = []
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

        # 处理每个试次
        one_session_all_trials_data = []
        one_session_all_trials_labels = []
        for trial_idx in range(1, dataset_info['num_trials']+1):
            # 获取当前trial在该文件中对应的变量，并进行分割
            matching_var = None
            for key in mat_data.keys():
                # 检查变量名是否是字符串类型
                if isinstance(key, str) and key.endswith(str(trial_idx)):
                    matching_var = key
                    break
            if matching_var:
                trial_data = mat_data[key]   
                dataset_labels = dataset_info['labels']                                                          # 获取当前trial的数据
                trial_label = dataset_labels[session_id-1][trial_idx-1]                                          # 获取当前trial的标签

                segemnt_trial_data, segemnt_trial_labels = segment_eeg_data(trial_data, trial_label, dataset_info['fs'], dataset_info['segment_len'])  # 使用封装的函数进行数据分割
                print(f"--trial{trial_idx}: data.shape: {trial_data.shape}, label: {trial_label} ==> 分割后: {segemnt_trial_data.shape[0]}个片段，数据形状: {segemnt_trial_data.shape}，标签形状: {segemnt_trial_labels.shape}")

                one_session_all_trials_data.append(segemnt_trial_data)
                one_session_all_trials_labels.append(segemnt_trial_labels)

        # 在样本维度上连接所有试次的数据和标签
        one_session_data = np.concatenate(one_session_all_trials_data, axis=0)
        one_session_labels = np.concatenate(one_session_all_trials_labels, axis=0)
        print(f"--会话{session_id}处理后的数据形状: {one_session_data.shape}，标签形状: {one_session_labels.shape}")
        
        # 添加到总列表中
        one_subject_all_sessions_data.append(one_session_data)
        one_subject_all_sessions_labels.append(one_session_labels)

    # 确保有数据可处理
    if not one_subject_all_sessions_data:
        print(f"被试 {subject_id} 没有可用数据")
        return False
    
    # 在N维度上连接所有会话的数据
    one_subject_all_sessions_data = np.concatenate(one_subject_all_sessions_data, axis=0)
    one_subject_all_sessions_labels = np.concatenate(one_subject_all_sessions_labels, axis=0)
    print(f"所有会话处理后的数据形状: {one_subject_all_sessions_data.shape}，标签形状: {one_subject_all_sessions_labels.shape}")
    
    # 创建字典保存数据和标签
    one_subject_data = {
        'data': one_subject_all_sessions_data,
        'label': one_subject_all_sessions_labels
    }
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"sub{subject_id}.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(one_subject_data, file)
    print(f"被试 {subject_id} 的数据已保存到: {file_path}")
    return True

def process_all_subjects(data_path, save_path, dataset):
    """
    处理所有被试的EEG数据
    
    参数:
    - root_path: 数据的根路径
    - dataset: 数据集名称
    - subject_ids: 被试ID列表
    
    返回:
    - 处理后的所有被试数据和标签
    """

    dataset_info = get_dataset_info(dataset)
    subject_ids = list(range(1, dataset_info['num_subjects'] + 1))
    print(f"待处理的被试ID: {subject_ids}")
    
    for subject_id in subject_ids:
        print(f"\n开始处理被试{subject_id}")
        success = segment_per_subject(data_path, save_path, dataset, subject_id)
        if not success:
            print(f"被试{subject_id}处理失败")
    
    print("\n所有被试处理完成")

if __name__ == "__main__":

    root_path = "/home/wuchennan/Dataset"
    dataset = ["seed3", "seed4", "seed5"][1]            # 选择数据集
    data_type = ["eeg_preprocessed", "eeg_feature"][0]     # 选择数据类型
    
    data_path = os.path.join(root_path, "raw_data", dataset, data_type)
    save_path = os.path.join(root_path, "segment_data", dataset, data_type)
    
    # 处理所有被试的数据
    process_all_subjects(data_path, save_path, dataset)
