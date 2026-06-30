"""
训练 YOLO 模型识别游戏中的 ball 和 paddle
需先用 collect_training_data.py 采集约 200+ 帧标注数据
"""

import os, yaml
from ultralytics import YOLO

# 生成数据集配置
data_cfg = {
    "path": os.path.abspath("dataset"),
    "train": "images",
    "val": "images",
    "names": {0: "ball", 1: "paddle"}
}
with open("dataset/data.yaml", "w") as f:
    yaml.dump(data_cfg, f, allow_unicode=True)

# 用 YOLOv8n（最小最快）fine-tune
model = YOLO("yolov8n.pt")
model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="boot_game",
    project="runs",
    patience=10,
)

print("\n训练完成！将 runs/boot_game/weights/best.pt 复制为 boot_game.pt")
print("然后在 play_boot.py 中设置 USE_YOLO = True")
