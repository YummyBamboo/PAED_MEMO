import libreface
import torch
import math


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
    m = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
         ]
    c = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
         ]
    k = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
         ]
    #x_t is the same size of au_sequences but fill with zeros
    x_t,damp_ratio,natural_freq,damp_freq = au_sequences.zeros_like

    t = frame_time
    for i in range(len(au_sequences)):
        damp_ratio[i] = c[i] / 2* math.sqrt((m[i] * k[i]))
        natural_freq[i] = math.sqrt(k[i] / m[i])
        damp_freq[i] = natural_freq[i] * math.sqrt(1-damp_ratio[i]**2)

        x_t[i] = au_sequences[i] * math.exp(-natural_freq[i]*damp_ratio[i]*t)*math.cos(damp_freq[i]*t) + force/k[i]


    #inertia constraints
    v_t = au_sequences.zeros_like
    for i in range(len(au_sequences)):
        v_t[i] = math.




