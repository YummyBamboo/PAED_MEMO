import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import ConvexHull

# 初始化MediaPipe面部网格模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

# --------------------- MediaPipe 468关键点映射表 ---------------------
# 根据LibreFace论文《Facial Action Unit Detection With Hybrid Representation Learning》调整
ANATOMY_MAPPING = {
    # ----------------- 上半面部AU -----------------
    "AU1": {  # 眉毛内侧提升
        "name": "Inner Brow Raiser",
        "indices": [70, 63, 105, 66, 107, 336, 296, 334],  # 眉间区域+内侧眼轮匝肌
        "roi_type": "ellipse",
        "scale": 0.85,  # 基于LibreFace的ROI缩放因子
        "color": (255, 50, 50)
    },
    "AU2": {  # 眉毛外侧提升
        "name": "Outer Brow Raiser",
        "indices": [46, 52, 53, 65, 55, 276, 283, 282],  # 颞肌附着区
        "roi_type": "ellipse",
        "scale": 1.15,
        "color": (50, 255, 50)
    },
     "AU4": {
        "name": "Brow Lowerer",
        "indices": list(range(105, 135)) + [333, 334, 336],  # 精确的皱眉肌区域
        "roi_type": "ellipse",
        "scale": 1.15,
        "color": (0, 0, 255)
    },
    # ----------------- 眼部AU -----------------
    "AU6": {  # 脸颊提升
        "name": "Cheek Raiser",
        "indices": [116, 117, 118, 119, 100, 47, 126, 209, 48],  # 眼轮匝肌外眦部
        "roi_type": "ellipse",
        "scale": 1.35,  # 覆盖颧大肌运动范围
        "color": (255, 255, 100)
    },
    "AU7": {  # 眼睑紧缩
        "name": "Lid Tightener",
        "indices": [160, 159, 158, 144, 145, 153],  # 上下眼睑关键点
        "roi_type": "rectangle",
        "aspect_ratio": 2.2,  # 水平矩形符合眼睑形态
        "color": (180, 180, 255)
    },
    # ----------------- 鼻部AU -----------------
    "AU9": {  # 鼻梁皱缩
        "name": "Nose Wrinkler",
        "indices": [193, 168, 417, 351, 419, 197],  # 鼻肌横部+降眉间肌
        "roi_type": "ellipse",
        "scale": 0.95,
        "color": (128, 0, 128)
    },
    # ----------------- 口部AU -----------------
    "AU10": {  # 上唇提升
        "name": "Upper Lip Raiser",
        "indices": [205, 206, 207, 36, 426, 216],  # 提上唇鼻翼肌区
        "roi_type": "ellipse",
        "scale": 1.1,
        "color": (255, 150, 50)
    },
    "AU12": {  # 嘴角外拉
        "name": "Lip Corner Puller",
        "indices": [308, 415, 310, 311, 312, 13, 82, 87, 317],  # 颧大肌附着区
        "roi_type": "ellipse",
        "scale": 1.45,  # 覆盖从耳前到嘴角的肌肉路径
        "color": (50, 255, 255)
    },
    "AU14": {  # 鼻唇沟加深
        "name": "Dimpler",
        "indices": [166, 75, 60, 20, 238, 79],  # 笑肌+大齿肌区
        "roi_type": "rectangle",
        "aspect_ratio": 0.7,  # 垂直矩形符合鼻唇沟走向
        "color": (150, 150, 255)
    },
    "AU15": {  # 嘴角下压
        "name": "Lip Corner Depressor",
        "indices": [378, 379, 416, 361, 323, 365, 291],  # 降口角肌区
        "roi_type": "ellipse",
        "scale": 1.05,
        "color": (255, 50, 150)
    },
    "AU17": {
        "name": "Chin Raiser",
        "indices": [200, 201, 202, 203, 204, 420, 421, 422],  # 精确的下巴运动区域
        "roi_type": "ellipse",
        "scale": 0.75,  # 小范围精确覆盖
        "color": (0, 128, 255)
    },
    "AU23": {  # 嘴唇紧缩
        "name": "Lip Tightener",
        "indices": [0, 17, 18, 37, 39, 40, 178, 405, 314],  # 口轮匝肌环形区
        "roi_type": "ellipse",
        "scale": 0.75,  # 小范围精确覆盖
        "color": (255, 128, 128)
    }
}


# --------------------- 增强版ROI生成算法 ---------------------
def generate_ellipse_roi(landmarks, center_indices, img_shape, scale=1.0):
    """支持多中心点的动态椭圆生成"""
    h, w = img_shape[:2]
    center_points = landmarks[center_indices]
    center = np.mean(center_points, axis=0)

    # 动态长轴计算（基于人脸尺寸标准化）
    face_width = np.linalg.norm(landmarks[454] - landmarks[234])  # 左右太阳穴间距
    axis_major = face_width * 0.3 * scale
    axis_minor = axis_major * 0.5

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    return cv2.ellipse(mask, tuple(center.astype(int)),
                       (int(axis_major), int(axis_minor)), 0, 0, 360, 255, -1)


def generate_polygon_roi(landmarks, vertex_indices, img_shape, expand=3):
    """带平滑处理的多边形生成"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    points = landmarks[vertex_indices]

    # 凸包计算与高斯平滑
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    smoothed = cv2.GaussianBlur(hull_points.astype(np.float32), (5, 5), 0)

    # 多边形扩展
    expanded = cv2.convexHull(smoothed) + np.random.uniform(-expand, expand, size=hull_points.shape)

    # 确保点集是整数类型
    expanded = expanded.astype(np.int32)

    # 检查点集的有效性
    if len(expanded) >= 3:
        cv2.fillConvexPoly(mask, expanded, 255)
    else:
        print("Error: Not enough points to form a polygon.")

    return mask


def generate_rectangle_roi(landmarks, region_indices, img_shape, aspect=1.0):
    """自适应宽高比矩形生成"""
    h, w = img_shape[:2]
    points = landmarks[region_indices]

    # PCA主成分分析确定方向
    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)
    _, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*vecs[:, 1]))

    # 旋转校正后的矩形
    rect = cv2.minAreaRect(points.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)
    return mask


# --------------------- 主流程 ---------------------
def process_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # MediaPipe检测
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise ValueError("No face detected")

    # 提取关键点坐标（归一化转像素坐标）
    landmarks = np.array([
        (lm.x * w, lm.y * h)
        for lm in results.multi_face_landmarks[0].landmark
    ])

    # 创建一个字典来存储每个AU的ROI掩码
    roi_masks = {}

    # 生成所有AU的ROI
    for au, config in ANATOMY_MAPPING.items():
        try:
            if config["roi_type"] == "ellipse":
                roi = generate_ellipse_roi(landmarks, config["indices"], image.shape, config["scale"])
            elif config["roi_type"] == "polygon":
                roi = generate_polygon_roi(landmarks, config["indices"], image.shape, config["expand_pixels"])
            elif config["roi_type"] == "rectangle":
                roi = generate_rectangle_roi(landmarks, config["indices"], image.shape, config["aspect_ratio"])

            # 存储ROI掩码到字典
            roi_masks[au] = roi

        except Exception as e:
            print(f"Error processing {au}: {str(e)}")

    return roi_masks


# 执行示例
roi_masks = process_image("E:\Code For Pytorch\Memo\\assets\examples\\face.png")

# 示例：访问AU4的ROI掩码
au4_roi = roi_masks.get("AU4")
if au4_roi is not None:
    cv2.imshow("AU4 ROI", au4_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
