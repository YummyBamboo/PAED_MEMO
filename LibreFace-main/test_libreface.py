import libreface
import time
import torch
import json

# inference on single image and store results to a variable
detected_facial_attributes = libreface.get_facial_attributes_image('E:\Code For Pytorch\Memo\\assets\examples\\face3.png',
                                                                   temp_dir = "./temp",
                                                                   device = "cuda")
print(detected_facial_attributes)

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



# inference on a single image and save results in a csv file
libreface.save_facial_attributes_image(image_path = "E:\Code For Pytorch\Memo\\assets\examples\\face3.png",
                                       output_save_path = "sample_image_results3.csv",
                                       temp_dir = "./temp",
                                       device = "cuda")

# inference on a video and store the results to a pandas dataframe
'''detected_facial_attributes_df = libreface.get_facial_attributes_video(video_path = "sample_disfa.avi",
                                                                      temp_dir = "./temp",
                                                                      device = "cpu")

# ## inference on a video and save the results framewise in a csv file
libreface.save_facial_attributes_video(video_path = "sample_disfa.avi",
                                       output_save_path = "sample_video_results.csv",
                                       temp_dir = "./temp",
                                       device = "cpu")

## inference on any image or video type and store results accordingly to a variable or save results
detected_facial_attributes = libreface.get_facial_attributes(file_path = "sample_disfa.avi",
                                                             output_save_path = "sample_results.csv",
                                                             temp_dir = "./temp",
                                                             device = "cpu")
                                                             
'''