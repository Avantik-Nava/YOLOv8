from exps.default import YOLOv8Exp


class Exp(YOLOv8Exp):
    def __init__(self):
        super().__init__()
        self.version = "n"
        self.exp_name = "yolov8_n"
