import math
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


    return au_features_tensor



def physical_model(au_sequences,force,frame_time):
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
        delta_au_intensity + au_sequences

    '''
    num_aus = len(au_sequences)
    m = [0.01] * num_aus
    c = [0.01] * num_aus
    k = [0.01] * num_aus

    # Initialize variables with appropriate shapes
    x_t = torch.zeros_like(au_sequences)
    damp_ratio = torch.zeros_like(au_sequences)
    natural_freq = torch.zeros_like(au_sequences)
    damp_freq = torch.zeros_like(au_sequences)

    t = frame_time
    for i in range(num_aus):
        damp_ratio[i] = c[i] / (2 * math.sqrt(m[i] * k[i]))
        natural_freq[i] = math.sqrt(k[i] / m[i])
        damp_freq[i] = natural_freq[i] * math.sqrt(1 - damp_ratio[i] ** 2)

        x_t[i] = au_sequences[i] * math.exp(-natural_freq[i] * damp_ratio[i] * t) * math.cos(damp_freq[i] * t) + force / k[i]
        x_t[i] = x_t[i] - au_sequences[i]

    return x_t




