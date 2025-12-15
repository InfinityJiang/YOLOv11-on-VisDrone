# 主函数

from ultralytics import YOLO
import cv2


def train_yolo_model():
    # 创建或加载模型 (这里使用预训练权重开始训练)
    model = YOLO("yolo11n.pt")  # 或使用 "yolo11n.yaml" 从头创建

    # 参数设置
    train_args = {
        "data": "VisDrone.yaml",  # 数据集配置文件（路径需正确）
        "epochs": 100,  # VisDrone需较长训练（小目标难收敛）
        "imgsz": 640,  # 保持高分辨率以检测小目标
        "batch": 12,  # 平衡显存和稳定性（RTX 3060级别GPU）
        "device": "0",  # 使用GPU加速
        "optimizer": "AdamW",  # AdamW优化器
        "lr0": 0.001,  # AdamW的经典初始学习率
        "weight_decay": 0.05,  # 防止过拟合
        "fliplr": 0.5,  # 水平翻转增强（对无人机场景有效）
        "mosaic": 1.0,  # 开启以提升小目标检测
        "close_mosaic": 10,  # 最后10轮关闭mosaic稳定训练
        "label_smoothing": 0.1,  # 缓解类别不平衡（VisDrone需此）
        "seed": 42,  # 固定随机种子（复现实验）
    }

    # 模型训练
    model.train(**train_args)
    print(f"训练完成")


def detect_image(model_path, image_path, output_path=None, conf_threshold=0.25):
    # 使用训练好的YOLO模型检测单张图像

    # 加载训练好的模型
    model = YOLO(model_path)

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 进行检测
    results = model.predict(
        source=img, conf=conf_threshold, save=output_path is not None, project="runs/detect", name="predict", exist_ok=True
    )

    # 获取带标注的图像
    annotated_img = results[0].plot()  # 获取第一张图像的标注结果

    # 保存到指定路径
    if output_path:
        cv2.imwrite(output_path, annotated_img)
        print(f"检测结果已保存到: {output_path}")

    return results, annotated_img


def test_yolo_model(model_path="path/to/best.pt"):
    # 在测试集上评估YOLO模型性能

    # 加载训练好的模型
    model = YOLO(model_path)

    # 测试参数配置
    test_args = {
        "data": "VisDrone.yaml",  # 数据集配置文件
        "split": "test",  # 测试集
        "batch": 12,  # 平衡显存和稳定性
        "save_json": True,  # 保存JSON
        "save_conf": True  # 保存置信度
    }

    # 运行测试
    model.val(**test_args)


if __name__ == "__main__":
    # train_yolo_model() # 目前模型已经训练完毕，若要训练，去掉这个注释
    model_path = r"runs\detect\train1\weights\best.pt"  # 模型路径，若经过额外的训练可进行替换
    # test_yolo_model(model_path)  # 对模型进行评估
    test_image_path = r"..\datasets\VisDrone\VisDrone2019-DET-train\images\0000076_04999_d_0000018.jpg"
    detect_image(model_path, test_image_path, 'result1.jpg')  # 对图片进行检测
