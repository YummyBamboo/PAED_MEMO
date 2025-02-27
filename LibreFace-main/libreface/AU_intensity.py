import torch
import libreface


def AU_intensity_detection(image_path):
    # inference on single image and store results to a variable
    detected_facial_attributes = libreface.get_facial_attributes_image(
    image_path,
        temp_dir="../temp",
        device="cuda")


# 提取 detected_aus 和 au_intensities
    detected_aus = detected_facial_attributes.get('detected_aus', {})
    au_intensities = detected_facial_attributes.get('au_intensities', {})

# 合并并转换为列表
    au_features = []
    for au, intensity in au_intensities.items():
        au_features.append(intensity)

    # 将 AU 特征和强度转换为 PyTorch 张量
    au_features_tensor = torch.tensor(au_features).unsqueeze(0)  # [1, T, 17]

    # 将张量转换为列表
    au_features_list = au_features_tensor.tolist()
    print(au_features_list)

    return au_features_list


