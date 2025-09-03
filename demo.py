import os
import cv2
import sys
import time

from elements.yolo import YOLO
from elements.deep_sort import DEEPSORT


# Load models
car_detector = YOLO(model_path="weights/best-car.engine", conf_thres=0.5, iou_thres=0.5, imgsz=(640, 640), max_det=12, device='0', classes={0: 'car'})
armor_detector = YOLO(model_path="weights/best-armor.engine", conf_thres=0.5, iou_thres=0, imgsz=(640, 640), max_det=2, device='0', classes={0: 'B1', 1: 'B2', 2: 'B3', 3: 'B4', 4: 'B5', 5: 'B7', 6: 'R1', 7: 'R2', 8: 'R3', 9: 'R4', 10: 'R5', 11: 'R7'})
deep_sort = DEEPSORT("configs/deep_sort.yaml")

# Video capture
source = 1
cap = cv2.VideoCapture(source)
print(f"Video source: {source}")

frame_count = int(cap.get(7))
print(f"Total number of frames: {frame_count}")
fps = int(cap.get(5))
fps = fps if fps else 30
print(f"FPS: {fps}")

w = int(cap.get(3))
h = int(cap.get(4))
print(f"Video resolution: {w}x{h}")

frame_num = 0

# Save output
output_name = "out_camera." if isinstance(source, int) else 'out_' + source.split('/')[-1].replace(source.split('.')[-1], "")
output_dir = os.path.join(os.getcwd(), 'inference\output')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_name + 'mp4')
print(f"Output path: {output_path}")

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

def plot_one_box(c1, c2, img, color, label, line_thickness):
    # Plots one bounding box on image img
    cv2.rectangle(img, c1, c2, color, line_thickness, cv2.LINE_AA)

    font_thickness = max(line_thickness - 1, 1)  # font thickness
    text_size = cv2.getTextSize(label, 0, line_thickness / 3, font_thickness)[0]
    c2 = c1[0] + text_size[0], c1[1] - text_size[1] - 3

    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], font_thickness, cv2.LINE_AA)

cost = time.time()
while cap.isOpened():
    frame_num += 1

    ret, frame = cap.read()

    if ret:
        print(f"\nFrame: {frame_num}")
        pic = frame.copy()

        start_time = time.time()
        car_output = car_detector.detect(frame)
        if car_output:
            for Car in car_output:
                print(f"Object: {Car}")
                xyxy = [Car['bbox'][0][0], Car['bbox'][0][1], Car['bbox'][1][0], Car['bbox'][1][1]]
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = xyxy[3]
                coords = (int(x_center), int(y_center))

                armor_output = armor_detector.detect(frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
                if armor_output:
                    for Armor in armor_output:
                        print(f"Object: {Armor}")
                        if 'R' in Armor['label']:
                            plot_one_box((Armor['bbox'][0][0] + xyxy[0], Armor['bbox'][0][1] + xyxy[1]), (Armor['bbox'][1][0] + xyxy[0], Armor['bbox'][1][1] + xyxy[1]), pic, (0, 0, 255), Armor['label'], 1)
                        elif 'B' in Armor['label']:
                            plot_one_box((Armor['bbox'][0][0] + xyxy[0], Armor['bbox'][0][1] + xyxy[1]), (Armor['bbox'][1][0] + xyxy[0], Armor['bbox'][1][1] + xyxy[1]), pic, (255, 0, 0), Armor['label'], 1)
                    if 'R' in Armor['label']:
                        plot_one_box((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), pic, (0, 0, 255), Armor['label'], 3)
                    elif 'B' in Armor['label']:
                        plot_one_box((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), pic, (255, 0, 0), Armor['label'], 3)
                else:
                    plot_one_box((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), pic, (0, 255, 0), Car['label'], 3)
                print(f"Coords: {coords}")
        end_time = time.time()

        cost = time.time() - cost
        cv2.putText(pic, f"FPS:{int(1 / cost)}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', pic)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Saving the output
        out.write(pic)

        sys.stdout.write(
            "\r[Input Video: %s] [%d/%d Frames Processed] [Time: %.1fms]\n"
            % (
                source,
                frame_num,
                frame_count,
                (end_time - start_time) * 1000
            )
        )
        print(cost*1000)

        cost = time.time()
    else:
        break

print(f'\n\nOutput video has been saved in {output_path}!')

cap.release()
cv2.destroyAllWindows()
