from deepface import DeepFace


def visual_emotion_detection(image_path):
    """
    使用DeepFace检测图像中的情感

    Args:
        image_path: 图像路径

    Returns:
        dominant_emotion: 主要情感类别
        dominant_confidence: 置信度
        emotions: 所有情感类别的置信度字典
    """

    analyze = DeepFace.analyze(img_path=image_path, actions=['emotion'])
    emotions = analyze[0]['emotion']
    dominant_emotion = analyze[0]['dominant_emotion']
    dominant_confidence = emotions[dominant_emotion]
    return dominant_emotion, dominant_confidence, emotions



