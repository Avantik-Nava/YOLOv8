#!/usr/bin/env python3
# YOLOv8 demo — exact YOLOX tools/demo.py CLI.
#
#   python tools/demo.py image -f exps/default/yolov8_s.py -c <ckpt> --path image.jpg --conf 0.25 --save_result
#   python tools/demo.py video -f <exp> -c <ckpt> --path video.mp4 --conf 0.25

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolox.data import letterbox  # noqa: E402
from yolox.exp import get_exp_by_file, get_exp_by_name  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser("YOLOv8 demo parser")
    parser.add_argument("demo", default="image", help="demo type, eg image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./assets/dog.jpg", type=str, help="path to image/video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true", help="save image/video result")
    parser.add_argument("-f", "--exp_file", default=None, type=str)
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="cpu", type=str, help="device to run model")
    parser.add_argument("--conf", default=0.25, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test image size")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser


def draw(img, dets, names):
    for x1, y1, x2, y2, conf, cls in dets.tolist():
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{names[int(cls)] if int(cls) < len(names) else int(cls)} {conf:.2f}"
        cv2.putText(img, label, (int(x1), max(int(y1) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def run_image(model, exp, path, conf, nms, tsize, save_result):
    import numpy as np
    img0 = cv2.imread(path)
    assert img0 is not None, f"not found: {path}"
    h0, w0 = img0.shape[:2]
    rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    lb, s, (dw, dh) = letterbox(rgb, tsize)
    t = torch.from_numpy(lb.transpose(2, 0, 1)).float().unsqueeze(0).to(next(model.parameters()).device) / 255.0
    t0 = time.time()
    dets = model.predict(t, conf_threshold=conf, iou_threshold=nms)[0].cpu()
    print(f"{len(dets)} dets in {(time.time()-t0)*1000:.1f}ms")
    if len(dets):
        dets[:, [0, 2]] = (dets[:, [0, 2]] - dw) / s
        dets[:, [1, 3]] = (dets[:, [1, 3]] - dh) / s
    names = getattr(exp, "names", None) or [str(i) for i in range(exp.num_classes)]
    out = draw(img0, dets, names)
    if save_result:
        cv2.imwrite("demo_result.jpg", out)
        print("saved demo_result.jpg")
    return out


def run_video(model, exp, path, conf, nms, tsize, save_result):
    cap = cv2.VideoCapture(0 if path == "webcam" else path)
    vw = None
    if save_result:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter("demo_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    names = getattr(exp, "names", None) or [str(i) for i in range(exp.num_classes)]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h0, w0 = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lb, s, (dw, dh) = letterbox(rgb, tsize)
        t = torch.from_numpy(lb.transpose(2, 0, 1)).float().unsqueeze(0).to(next(model.parameters()).device) / 255.0
        dets = model.predict(t, conf_threshold=conf, iou_threshold=nms)[0].cpu()
        if len(dets):
            dets[:, [0, 2]] = (dets[:, [0, 2]] - dw) / s
            dets[:, [1, 3]] = (dets[:, [1, 3]] - dh) / s
        frame = draw(frame, dets, names)
        if save_result and vw is not None:
            vw.write(frame)
        cv2.imshow("yolov8 demo (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()


def main():
    args = make_parser().parse_args()
    if args.exp_file is not None:
        exp = get_exp_by_file(args.exp_file)
    elif args.name is not None:
        exp = get_exp_by_name(args.name)
    else:
        exp = get_exp_by_file("exps/default/yolov8_s.py")
    exp.merge(args.opts)
    print(exp)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = exp.get_model().to(device).eval()
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"loaded {args.ckpt}")

    if args.demo in ("image", "webcam"):
        path = "webcam" if args.demo == "webcam" else args.path
        if args.demo == "webcam":
            run_video(model, exp, path, args.conf, args.nms, args.tsize, args.save_result)
        else:
            run_image(model, exp, path, args.conf, args.nms, args.tsize, args.save_result)
    elif args.demo == "video":
        run_video(model, exp, args.path, args.conf, args.nms, args.tsize, args.save_result)


if __name__ == "__main__":
    main()
