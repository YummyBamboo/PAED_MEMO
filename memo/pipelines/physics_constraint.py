import math
import torch
import libreface
from deepface import DeepFace


def AU_intensity_detection(image_path):
    # inference on single image and store results to a variable
    detected_facial_attributes = libreface.get_facial_attributes_image(
    image_path,
        temp_dir="../temp",
        device="cpu")

# 提取 detected_aus 和 au_intensities
    detected_aus = detected_facial_attributes.get('detected_aus', {})
    au_intensities = detected_facial_attributes.get('au_intensities', {})


# 合并并转换为列表
    au_features = []
    for au, intensity in au_intensities.items():
        au_features.append(intensity)

    # 将 AU 特征和强度转换为 PyTorch 张量
    au_features_tensor = torch.tensor(au_features).unsqueeze(0)
    au_features_tensor = au_features_tensor.to('cuda')

    return au_features_tensor






def physical_model(au_sequences, force, frame_time):
    # damped harmonic system
        '''

    Args:
        au_sequences: the au_intensity
        m: the inertia of au  (determined by experimental results)
        c: fraction para  has differences between aus (determined by experimental results)
        k: recovery para
        force: driven force from audio info
        frame_time: time of the video based on frame

    Returns:
        delta_au_intensity

        '''


        au_sequences.squeeze()


    # 定义每个AU的参数
        m = torch.tensor([
        0.02, 0.015, 0.03, 0.025, 0.02,
        0.025, 0.02, 0.015, 0.025, 0.03,
        0.025, 0.02
        ], device=au_sequences.device)

        c = torch.tensor([
        0.05, 0.04, 0.06, 0.05, 0.04,
        0.06, 0.05, 0.04, 0.06, 0.07,
        0.06, 0.05
        ], device=au_sequences.device)

        k = torch.tensor([
        0.1, 0.08, 0.12, 0.1, 0.08,
        0.12, 0.1, 0.08, 0.12, 0.14,
        0.12, 0.1
        ], device=au_sequences.device)

    # 向量化计算（移除循环）
        damp_ratio = c / (2 * torch.sqrt(m * k))  # 阻尼比
        natural_freq = torch.sqrt(k / m)  # 自然频率
        damp_freq = natural_freq * torch.sqrt(1 - damp_ratio ** 2)  # 阻尼频率
        force = force/100

        t = frame_time
        decay = torch.exp(-natural_freq * damp_ratio * t)  # 衰减函数
        oscillation = torch.cos(damp_freq * t)  # 振荡函数

        x_t = au_sequences * decay * oscillation + force / k  # 最终的AU强度
        return x_t - au_sequences  # 返回值

