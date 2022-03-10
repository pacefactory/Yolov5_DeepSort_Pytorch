# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')
# from local.Yolov5_DeepSort_Pytorch.yolov5.utils.datasets import LoadImages
# from local.Yolov5_DeepSort_Pytorch.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
# from local.Yolov5_DeepSort_Pytorch.yolov5.utils.torch_utils import select_device, time_sync
# from local.Yolov5_DeepSort_Pytorch.deep_sort_pytorch.utils.parser import get_config
# from local.Yolov5_DeepSort_Pytorch.deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import time
from pathlib import Path
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                   xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_sync
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def detect(imgs, idxs, img_wh, opt):
    # Incoming change: yolo_weights and deep_sort_weights were replaced by yolo_model and deep_sort_model
    yolo_model, deep_sort_model, imgsz, half = \
         opt.yolo_model, opt.deep_sort_model, \
         opt.imgsz, opt.half,
    img_width, img_height = img_wh
    # The webcam was incoming, should not be needed
    # webcam = source == '0' or source.startswith(
    #     'rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(opt.device)
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )

    # Initializes
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
        
    # OLD: dataset = LoadImages(imgs, idxs, img_size=imgsz)
    # Incoming load images:
    dataset = LoadImages(imgs=imgs, idxs=idxs ,img_size=imgsz, stride=stride, auto=pt and not jit)
    dataset_length = len(dataset)

    # run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    t0 = time.time() # time variable from Jack's changes
    output_dict={}

    # for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset): # incoming changes
    for relative_frame, (img, im0s, ems, frame_idx) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(img, augment=opt.augment)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for det in pred:  # detections per image
            s, im0 = '', im0s
            seen += 1
            s += '%gx%g ' % img.shape[2:]  # print string


            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for (output, conf) in zip(outputs, confs):

                        ## Jacks changes 

                        id = int(output[4])

                        instance_dict={}
                        instance_dict['ems']=ems
                        instance_dict['frame-idx']=frame_idx

                        bbox={}
                        bbox['xmin'] = output[0]/img_width
                        bbox['ymax'] = (img_height - output[1])/img_height
                        bbox['xmax'] = output[2]/img_width
                        bbox['ymin'] = (img_height - output[3])/img_height
                        instance_dict['bbox']=bbox

                        confidence=float(conf)
                        instance_dict['confidence']=confidence
                        
                        if id not in output_dict:
                            obj_dict={}
                            obj_dict['start-and-end-times']=None
                            label= names[int(output[5])]
                            obj_dict['label']=label
                            color = compute_color_for_id(id)
                            obj_dict['colour']=color

                            frames_list=[None]*dataset_length
                            frames_list[relative_frame]=instance_dict
                            obj_dict['frames']=frames_list

                            output_dict[id]=obj_dict

                        else:
                            output_dict[id]['frames'][relative_frame]=instance_dict

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Print time (inference + NMS)
            print("Relative Frame: " + str(relative_frame))
            print("Frame Idx: " + str(frame_idx))
            print('%sDone. (%.3fs)' % (s, t2 - t1))

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print('Done. (%.3fs)' % (time.time() - t0))
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)

    return output_dict



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcams
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # class 0 is person, 1 is bicycle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    # old method
    # args.imgsz = check_img_size(args.imgsz)
    return args

# This may fail, I'm not sure if kwargs can be anything by the only param.
#     - check to see if parse_known_args()[0] refers to the nth parameter of track()
def track(imgs, idxs, img_wh, **kwargs):
    # Usage: import track; track.track(imgsz=320, yolo_weights='yolov5m.pt')
    args = parse_opt(True)
    for k, v in kwargs.items():
        setattr(args, k, v)
    with torch.no_grad():
        out = detect(imgs, idxs, img_wh, args)
        return out

if __name__ == '__main__':
    args=parse_opt()
    with torch.no_grad():
        detect(args )
