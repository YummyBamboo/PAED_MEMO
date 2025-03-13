import json
import random
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from torch.utils.data import Dataset
from torchvision import transforms
from memo.pipelines.physics_constraint import AU_intensity_detection
from memo.pipelines.AU_ROI_detection_mediapipe import AU_ROI_detection

class VideoDataset(Dataset):
    def __init__(
        self,
        num_past_frames,
        n_sample_frames,
        img_size=(512, 512),
        audio_margin=2,
        metadata_paths=None,
    ):
        super().__init__()
        self.num_past_frames = num_past_frames
        self.n_sample_frames = n_sample_frames
        self.img_size = img_size
        self.audio_margin = audio_margin

        self.metadata = []
        for metadata_path in metadata_paths:
            for line in Path(metadata_path).read_text().splitlines():
                self.metadata.append(json.loads(line))
        self.length = len(self.metadata)

        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(self.img_size[0], antialias=True),
                transforms.CenterCrop(self.img_size),
                transforms.Normalize([0.5], [0.5], inplace=True),
            ]
        )

    def get_batch(self, idx, drop_rate=0.05):
        video_info = self.metadata[idx]
        video_path = video_info["video"]
        face_emb_path = video_info["face_emb"]
        audio_emb_path = video_info["audio_emb"]
        audio_emotion_path = video_info["audio_emotion"]
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        assert video_length > self.n_sample_frames + self.num_past_frames + 2 * self.audio_margin

        start_idx = random.randint(
            self.num_past_frames,
            video_length - self.n_sample_frames - self.audio_margin - 1,
        )

        batch_idx = np.arange(start_idx, start_idx + self.n_sample_frames)
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_idx).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.0  # (f, c, h, w)

        ref_img_idx = random.randint(
            self.num_past_frames,
            video_length - self.n_sample_frames - self.audio_margin - 1,
        )

        pixel_values_ref_img = (
            torch.from_numpy(video_reader[ref_img_idx].asnumpy()).unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        )
        pixel_values_ref_img = pixel_values_ref_img / 255.0

        face_emb = torch.load(face_emb_path)  # (512)
        full_audio_emb = torch.load(audio_emb_path)  # (n, 12, 768)
        assert (
            full_audio_emb.shape[0] == video_length
        ), f"Audio embedding shape {full_audio_emb.shape[0]} != video length {video_length}"
        indices = torch.arange(2 * self.audio_margin + 1) - self.audio_margin  # [-2, -1, 0, 1, 2]

        center_indices = torch.arange(
            start_idx,
            start_idx + self.n_sample_frames,
        ).unsqueeze(
            1
        ) + indices.unsqueeze(0)
        audio_emb = full_audio_emb[center_indices]  # (f, 5, 12, 768)

        if torch.rand(1).item() < drop_rate:
            for i in range(self.audio_margin):
                audio_emb[i, : self.audio_margin - i] = audio_emb[i, self.audio_margin - i]

        full_audio_emotion = torch.load(audio_emotion_path)
        assert (
            full_audio_emotion.shape[0] == video_length
        ), f"Audio embedding shape {full_audio_emotion.shape[0]} != video length {video_length}"
        audio_emotion = full_audio_emotion[start_idx : start_idx + self.n_sample_frames]
        audio_emotion = torch.mode(audio_emotion).values.item()

        # 计算 AU intensities 和 masks
        au_intensities = []
        au_masks = []
        for frame in pixel_values:  # pixel_values 是 [0,1] 范围的张量
            # 反归一化到 [0,255] 并转为 PIL 图像
            frame_unnormalized = (frame * 0.5 + 0.5) * 255  # 反Normalize操作
            frame_unnormalized = frame_unnormalized.byte()  # 转为整数张量
            pil_frame = transforms.ToPILImage()(frame_unnormalized)

            # 计算 AU 强度和 ROI 掩码
            au_intensity = AU_intensity_detection(pil_frame)
            au_mask = AU_ROI_detection(pil_frame)

            au_intensities.append(au_intensity)
            au_masks.append(au_mask)

        # 转换为张量（假设 AU_intensity 是形状一致的 Tensor）
        au_intensities = torch.stack(au_intensities)  # 形状：(frame_num, num_AUs)
        # au_masks 可能需要根据具体返回格式处理（例如字典或张量）

        if self.num_past_frames > 0:
            batch_idx = np.arange(start_idx - self.num_past_frames, start_idx)
            pixel_values_motion = (
                torch.from_numpy(video_reader.get_batch(batch_idx).asnumpy()).permute(0, 3, 1, 2).contiguous()
            )
            pixel_values_motion = pixel_values_motion / 255.0  # (num_past_frames, c, h, w)

            if torch.rand(1).item() < drop_rate:
                pixel_values_motion = pixel_values[0].unsqueeze(0).repeat(self.num_past_frames, 1, 1, 1)

            pixel_values_ref_img = torch.cat([pixel_values_ref_img, pixel_values_motion], dim=0)

        del video_reader

        return (
            pixel_values,
            pixel_values_ref_img,
            face_emb,
            audio_emb,
            audio_emotion,
            au_intensities,
            au_masks
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                (
                    pixel_values,
                    pixel_values_ref_img,
                    face_emb,
                    audio_emb,
                    audio_emotion,
                    au_intensities,
                    au_masks
                ) = self.get_batch(idx)
                break
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                idx = random.randint(0, self.length - 1)

        # 应用像素变换（归一化和裁剪）
        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values_ref_img = self.pixel_transforms(pixel_values_ref_img)

        sample = {
            "pixel_values": pixel_values,
            "pixel_values_ref_img": pixel_values_ref_img,
            "face_emb": face_emb,
            "audio_emb": audio_emb,
            "audio_emotion": audio_emotion,
            "au_intensities": au_intensities,  # 新增字段
            "au_masks": au_masks,             # 新增字段
        }

        return sample
import json
import random
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from torch.utils.data import Dataset
from torchvision import transforms
from memo.pipelines.physics_constraint import AU_intensity_detection
from memo.pipelines.AU_ROI_detection_mediapipe import AU_ROI_detection

class VideoDataset(Dataset):
    def __init__(
        self,
        num_past_frames,
        n_sample_frames,
        img_size=(512, 512),
        audio_margin=2,
        metadata_paths=None,
    ):
        super().__init__()
        self.num_past_frames = num_past_frames
        self.n_sample_frames = n_sample_frames
        self.img_size = img_size
        self.audio_margin = audio_margin

        self.metadata = []
        for metadata_path in metadata_paths:
            for line in Path(metadata_path).read_text().splitlines():
                self.metadata.append(json.loads(line))
        self.length = len(self.metadata)

        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(self.img_size[0], antialias=True),
                transforms.CenterCrop(self.img_size),
                transforms.Normalize([0.5], [0.5], inplace=True),
            ]
        )

    def get_batch(self, idx, drop_rate=0.05):
        video_info = self.metadata[idx]
        video_path = video_info["video"]
        face_emb_path = video_info["face_emb"]
        audio_emb_path = video_info["audio_emb"]
        audio_emotion_path = video_info["audio_emotion"]
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        assert video_length > self.n_sample_frames + self.num_past_frames + 2 * self.audio_margin

        start_idx = random.randint(
            self.num_past_frames,
            video_length - self.n_sample_frames - self.audio_margin - 1,
        )

        batch_idx = np.arange(start_idx, start_idx + self.n_sample_frames)
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_idx).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.0  # (f, c, h, w)

        ref_img_idx = random.randint(
            self.num_past_frames,
            video_length - self.n_sample_frames - self.audio_margin - 1,
        )

        pixel_values_ref_img = (
            torch.from_numpy(video_reader[ref_img_idx].asnumpy()).unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        )
        pixel_values_ref_img = pixel_values_ref_img / 255.0

        face_emb = torch.load(face_emb_path)  # (512)
        full_audio_emb = torch.load(audio_emb_path)  # (n, 12, 768)
        assert (
            full_audio_emb.shape[0] == video_length
        ), f"Audio embedding shape {full_audio_emb.shape[0]} != video length {video_length}"
        indices = torch.arange(2 * self.audio_margin + 1) - self.audio_margin  # [-2, -1, 0, 1, 2]

        center_indices = torch.arange(
            start_idx,
            start_idx + self.n_sample_frames,
        ).unsqueeze(
            1
        ) + indices.unsqueeze(0)
        audio_emb = full_audio_emb[center_indices]  # (f, 5, 12, 768)

        if torch.rand(1).item() < drop_rate:
            for i in range(self.audio_margin):
                audio_emb[i, : self.audio_margin - i] = audio_emb[i, self.audio_margin - i]

        full_audio_emotion = torch.load(audio_emotion_path)
        assert (
            full_audio_emotion.shape[0] == video_length
        ), f"Audio embedding shape {full_audio_emotion.shape[0]} != video length {video_length}"
        audio_emotion = full_audio_emotion[start_idx : start_idx + self.n_sample_frames]
        audio_emotion = torch.mode(audio_emotion).values.item()

        # 计算 AU intensities 和 masks
        au_intensities = []
        au_masks = []
        for frame in pixel_values:  # pixel_values 是 [0,1] 范围的张量
            # 反归一化到 [0,255] 并转为 PIL 图像
            frame_unnormalized = (frame * 0.5 + 0.5) * 255  # 反Normalize操作
            frame_unnormalized = frame_unnormalized.byte()  # 转为整数张量
            pil_frame = transforms.ToPILImage()(frame_unnormalized)

            # 计算 AU 强度和 ROI 掩码
            au_intensity = AU_intensity_detection(pil_frame)
            au_mask_dict = AU_ROI_detection(pil_frame)

            mask_tensors = [v for k,v in au_mask_dict.items()]
            au_mask = torch.stack(mask_tensors)

            au_intensities.append(au_intensity)
            au_masks.append(au_mask)

        # 转换为张量（假设 AU_intensity 是形状一致的 Tensor）
        au_intensities = torch.stack(au_intensities)  # 形状：(frame_num, num_AUs)
        au_masks = torch.stack(au_masks)

        if self.num_past_frames > 0:
            batch_idx = np.arange(start_idx - self.num_past_frames, start_idx)
            pixel_values_motion = (
                torch.from_numpy(video_reader.get_batch(batch_idx).asnumpy()).permute(0, 3, 1, 2).contiguous()
            )
            pixel_values_motion = pixel_values_motion / 255.0  # (num_past_frames, c, h, w)

            if torch.rand(1).item() < drop_rate:
                pixel_values_motion = pixel_values[0].unsqueeze(0).repeat(self.num_past_frames, 1, 1, 1)

            pixel_values_ref_img = torch.cat([pixel_values_ref_img, pixel_values_motion], dim=0)

        del video_reader

        return (
            pixel_values,
            pixel_values_ref_img,
            face_emb,
            audio_emb,
            audio_emotion,
            au_intensities,
            au_masks
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                (
                    pixel_values,
                    pixel_values_ref_img,
                    face_emb,
                    audio_emb,
                    audio_emotion,
                    au_intensities,
                    au_masks
                ) = self.get_batch(idx)
                break
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                idx = random.randint(0, self.length - 1)

        # 应用像素变换（归一化和裁剪）
        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values_ref_img = self.pixel_transforms(pixel_values_ref_img)

        sample = {
            "pixel_values": pixel_values,
            "pixel_values_ref_img": pixel_values_ref_img,
            "face_emb": face_emb,
            "audio_emb": audio_emb,
            "audio_emotion": audio_emotion,
            "au_intensities": au_intensities,  # 新增字段
            "au_masks": au_masks,             # 新增字段
        }

        return sample
