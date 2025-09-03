import argparse


class ArgumentsBase(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def main_args_initialization(self):
        self.parser.add_argument('--device',
                            default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.parser.add_argument('--camera', type=int,
                            default=1, help='camera source')
        self.parser.add_argument('--map', type=str,
                            default='inference/map.png', help='map path')
        self.parser.add_argument('--size', type=int,
                            default=(2500, 1250), help='frame size')
        self.parser.add_argument('--car-model-path', type=str,
                            default='weights/best-car.engine', help='car model path')
        self.parser.add_argument('--armor-model-path', type=str,
                            default='weights/best-armor.engine', help='armor model path')
        self.parser.add_argument('--car-conf-thres', type=float,
                            default=0.5, help='car confidence threshold')
        self.parser.add_argument('--armor-conf-thres', type=float,
                            default=0.5, help='armor confidence threshold')
        self.parser.add_argument('--car-iou-thres', type=float,
                            default=0.5, help='car iou threshold')
        self.parser.add_argument('--armor-iou-thres', type=float,
                            default=0, help='armor iou threshold')
        self.parser.add_argument('--car-imgsz', type=int,
                            default=(640, 640), help='car image size')
        self.parser.add_argument('--armor-imgsz', type=int,
                            default=(640, 640), help='armor image size')
        self.parser.add_argument('--car-max-det', type=int,
                            default=12, help='car max detections')
        self.parser.add_argument('--armor-max-det', type=int,
                            default=2, help='armor max detections')
        self.parser.add_argument('--car-classes', type=str,
                            default={0: 'car'}, help='car classes')
        self.parser.add_argument('--armor-classes', type=str,
                            default={0: 'B1', 1: 'B2', 2: 'B3', 3: 'B4', 4: 'B5', 5: 'B7', 6: 'R1', 7: 'R2', 8: 'R3', 9: 'R4', 10: 'R5', 11: 'R7'}, help='armor classes')
        self.parser.add_argument('--map-scale', type=float,
                            default=1/3, help='map scale')

    def _parse(self):
        self.opt = self.parser.parse_args()

        return self.opt


class Arguments(ArgumentsBase):
    def __init__(self):
        super().__init__()
        self.main_args_initialization()

    def parse(self):
        opt = self._parse()

        return opt
