import torch.backends.cudnn as cudnn
import torch
import cv2
from pathlib import Path
import time
import shutil
import platform
import os
import argparse
from local.Yolov5_DeepSort_Pytorch.deep_sort_pytorch.deep_sort import DeepSort
from local.Yolov5_DeepSort_Pytorch.deep_sort_pytorch.utils.parser import get_config
from local.Yolov5_DeepSort_Pytorch.yolov5.utils.plots import plot_one_box
from local.Yolov5_DeepSort_Pytorch.yolov5.utils.torch_utils import select_device, time_sync
from local.Yolov5_DeepSort_Pytorch.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from local.Yolov5_DeepSort_Pytorch.yolov5.utils.datasets import LoadImages, LoadStreams
from local.Yolov5_DeepSort_Pytorch.yolov5.models.experimental import attempt_load
from local.Yolov5_DeepSort_Pytorch.yolov5.utils.downloads import attempt_download
import sys
sys.path.insert(0, './yolov5')


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def detect(imgs, idxs, img_wh, opt):
    yolo_weights, deep_sort_weights, imgsz, evaluate = \
        opt.yolo_weights, opt.deep_sort_weights, opt.img_size, opt.evaluate
    img_width, img_height = img_wh

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    # TODO: avoid this download by placing the weights in the home dir of forked repo
    #
    #      play with the weights // config HYPs once testing is set up (optimize for usecase)

    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # TODO: delete lines 58-64 (below here if --evaluate remains unused)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    #  if not evaluate:
    #      if os.path.exists(out):
    #          pass
    #          shutil.rmtree(out)  # delete output folder
    #      os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    dataset = LoadImages(imgs, idxs, img_size=imgsz)
    dataset_length = len(dataset)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    output_dict = {}

    for relative_frame, (img, im0s, ems, frame_idx) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        id = int(output[4])

                        instance_dict = {}
                        instance_dict['ems'] = ems
                        instance_dict['frame-idx'] = frame_idx

                        bbox = {}
                        # The old coordinate system:
                        # bbox['xmin'] = output[0]/img_width
                        # bbox['ymax'] = (img_height - output[1])/img_height
                        # bbox['xmax'] = output[2]/img_width
                        # bbox['ymin'] = (img_height - output[3])/img_height
                        bbox['xmin'] = output[0] / img_width
                        bbox['ymin'] = output[1] / img_height
                        bbox['xmax'] = output[2] / img_width
                        bbox['ymax'] = output[3] / img_height
                        instance_dict['bbox'] = bbox

                        confidence = float(conf)
                        instance_dict['confidence'] = confidence

                        if id not in output_dict:
                            obj_dict = {}
                            obj_dict['start-and-end-times'] = None
                            label = names[int(output[5])]
                            obj_dict['label'] = label
                            color = compute_color_for_id(id)
                            obj_dict['colour'] = color

                            frames_list = [None]*dataset_length
                            frames_list[relative_frame] = instance_dict
                            obj_dict['frames'] = frames_list

                            output_dict[id] = obj_dict

                        else:
                            output_dict[id]['frames'][relative_frame] = instance_dict

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print("Relative Frame: " + str(relative_frame))
            print("Frame Idx: " + str(frame_idx))
            print('%sDone. (%.3fs)' % (s, t2 - t1))

    print('Done. (%.3fs)' % (time.time() - t0))

    return output_dict


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str,
                        default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')

    # file/folder, 0 for webcam
    # TODO: figure out if a folder of images can be passed as opposed to a video (will avoid
    #      stitching together the snapshots, but may require saving the images in a file which may
    #      be suboptimal compared to the video stitch method)
    #      IDEALLY: lets see if we can just pass in the PIL images directly as that's likely what's
    #               happening behind the scenes anyway
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='MOT16 evaluation')
    parser.add_argument("--config_deepsort", type=str,
                        default="local/Yolov5_DeepSort_Pytorch/deep_sort_pytorch/configs/deep_sort.yaml")
    #  parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    return args


# TODO: this my fail, I'm not sure if kwargs can be anything but the only param
#     - check to see if parse_known_args()[0] refers to the nth parameter of track()
def track(imgs, idxs, img_wh, **kwargs):
    # Usage: import track; track.track(imgsz=320, yolo_weights='yolov5m.pt')
    args = parse_opt(True)
    for k, v in kwargs.items():
        setattr(args, k, v)
    with torch.no_grad():
        # TODO: this my fail, I'm not sure if kwargs can be anything but the only param
        out = detect(imgs, idxs, img_wh, args)
        return out


if __name__ == '__main__':
    args = parse_opt()
    with torch.no_grad():
        detect(args)
