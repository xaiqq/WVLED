import argparse
import os
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import platform
import cv2
from tqdm import tqdm
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (check_file, increment_path, check_img_size,
                            non_max_suppression, scale_boxes, save_one_box, 
                            colorstr, strip_optimizer)
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from models import build_model
from datasets.loader import LoadImages
from utils.plots import Annotator, colors
from utils.general import LOGGER
from configs.defaults import get_cfg
from utils.checkpoint import load_config
import utils.checkpoint as cu
from datasets import cv2_transform
from datasets.utils import get_sequence



IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

def inference_one(cfg,
                  weights,
                  video_path, 
                  model, 
                  names, 
                  save_dir,
                  conf_thres,
                  iou_thres,
                  classes,
                  agnostic_nms,
                  max_det,
                  save_crop,
                  save_img,
                  view_img,
                  line_thickness,
                  hide_conf,
                  hide_labels,
                  save_txt,
                  update,
                  device,
                  ):
    windows = []
    FPS = []
    crop_size = cfg.DATA.TEST_CROP_SIZE
    vid_path = [None]
    vid_writer = [None]
    cap = cv2.VideoCapture(video_path)
    cnt = 1
    queue = []
    s = ""
    images = []
    sample_rate = cfg.DATA.SAMPLING_RATE
    video_length = cfg.DATA.NUM_FRAMES
    seq_len = sample_rate * video_length
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        images.append(frame)
    cap.release()
    for key_frame, im0s in enumerate(images):
        # Add the read frame to last and pop out the oldest one
        seq = get_sequence(key_frame, seq_len // 2, sample_rate, len(images))
        imgs = []
        for seq_frame in seq:
            imgs.append(cv2_transform.resize(crop_size, images[seq_frame]))

        # frame = img = cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(cfg.DATA.MEAN, dtype=np.float32),
                np.array(cfg.DATA.STD, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        imgs = torch.unsqueeze(imgs, 0)

        imgs = imgs.to(device)
        # Model inference
        t1 = time.time()
        pred, _ = model(imgs)
        FPS.append(time.time() - t1)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0 = video_path, im0s.copy()
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += '%gx%g ' % imgs.shape[3:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(imgs.shape[3:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Print results
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    return FPS

@torch.no_grad()
def run(weights=ROOT / 'best.pyth',  # model.pt path(s)
        source=ROOT / 'data/JHMDB/videos',  # file/dir/URL/glob, 0 for webcam
        imgsz=256,  # inference size (pixels)
        data=ROOT / 'data/JHMDB/annotations/jhmdb_action_list.pbtxt',  # dataset.yaml path
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        cfg_files=["configs/JHMDB_base.yaml"],
        vid_stride=4,  # video frame-rate stride
        ):
    cfg = get_cfg()
    for path_to_config in cfg_files:
        cfg.merge_from_file(path_to_config)
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = build_model(cfg)
    cu.load_checkpoint(weights, model, False)
    model = model.to(device)
    model.eval()
    names = dict(model.names.items() if hasattr(model, 'names') else model.module.names.items())
    videos_path = os.listdir(source)
    pbar = tqdm()
    FPS_all = []
    for video_path in tqdm(videos_path):
        video_path = os.path.join(source, video_path)
        FPS = inference_one(cfg, weights, video_path, model, names, 
                    save_dir, conf_thres, iou_thres, classes,
                    agnostic_nms, max_det, save_crop, save_img, view_img,
                    line_thickness, hide_conf, hide_labels, save_txt, update, device)
        FPS_all += FPS
    print('平均推理帧率:',1 / np.mean(FPS_all))
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/val/exp/weight/best.pyth', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/source_videos/CliffDiving', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument("--cfg", dest="cfg_files", help="Path to the config files", default=["configs/UCF101_24_base_detect.yaml"], nargs="+")
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[256], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    run(**vars(opt))
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)