import libreface
import torch

# 物理约束参数
physical_constraint_interval = 5
physical_constraint_strength = 0.1

# 获取AU特征
def get_AU_features(image_path):

    detected_facial_attributes = libreface.get_facial_attributes_image(image_path,                                                       temp_dir = "./temp",
                                                                   device = "cuda")
    # 提取 detected_aus 和 au_intensities
    detected_aus = detected_facial_attributes.get('detected_aus', {})
    au_intensities = detected_facial_attributes.get('au_intensities', {})

    print(detected_aus)

    # 合并并转换为列表
    au_features = []
    for au, intensity in au_intensities.items():
        au_features.append(intensity)

    # 将 AU 特征和强度转换为 PyTorch 张量
    au_features_tensor = torch.tensor(au_features, dtype=torch.float).unsqueeze(0)  # [1, T, 17]

    print(au_features_tensor.size())

    # 将张量转换为列表
    au_sequences = au_features_tensor.tolist()
    return au_sequences
