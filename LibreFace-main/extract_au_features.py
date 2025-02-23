import sys
import libreface
import torch
import cv2
import os
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_au_features.py [input_path]")
        sys.exit(1)

    input_path = sys.argv[1]
    temp_dir = "./temp"
    device = "cpu"

    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at {input_path}")
            sys.exit(1)

        print(f"Processing input file: {input_path}")

        detected_facial_attributes = libreface.get_facial_attributes_image(
            image_path=input_path,
            temp_dir=temp_dir,
            device=device
        )

        # 打印检测到的面部属性
        print("Detected Facial Attributes:", detected_facial_attributes)

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

        # 输出为 JSON 格式
        print(json.dumps(au_features_list))

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()