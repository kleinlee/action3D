import os
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image


def main():
    import argparse
    import glob
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SMPLest_X Demo')
    parser.add_argument('video_paths', type=str, help='视频文件路径或通配符模式')
    args = parser.parse_args()
    
    # 处理输入的视频路径
    video_paths = glob.glob(args.video_paths)
    cudnn.benchmark = True
    ckpt_name = 'smplest_x_h'

    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models/smplest_x_h/config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models/smplest_x_h/smplest_x_h.pth.tar')
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    exp_name = f'inference_{ckpt_name}_{time_str}'
    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', "xxx", 'log'),  
        }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()

    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)
    transform = transforms.ToTensor()
    for video_path in video_paths:
        results = []
        output_file = video_path.replace('.mp4', '.json')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_index in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            original_img = frame[:, :, ::-1].copy().astype(np.float32)
            original_img_height, original_img_width = original_img.shape[:2]
        
            # detection, xyxy
            yolo_bbox = detector.predict(original_img, 
                                device='cuda', 
                                classes=00, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

            box = yolo_bbox[0]
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = box[0]
            yolo_bbox_xywh[1] = box[1]
            yolo_bbox_xywh[2] = abs(box[2] - box[0])
            yolo_bbox_xywh[3] = abs(box[3] - box[1])
            
            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=cfg.model.input_img_shape, 
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
            img, _, _ = generate_patch_image(cvimg=original_img, 
                                                bbox=bbox, 
                                                scale=1.0, 
                                                rot=0.0, 
                                                do_flip=False, 
                                                out_shape=cfg.model.input_img_shape)
                
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')
            
            # 打印out中的元素
            print(out.keys())
            for key in out.keys():
                print(key, out[key].shape)

            # save results for this frame
            frame_result = {
                'frame': frame_index,
                'smplx_root_pose': out['smplx_root_pose'].detach().cpu().numpy()[0].tolist(),
                'smplx_body_pose': out['smplx_body_pose'].detach().cpu().numpy()[0].tolist(),
                'smplx_lhand_pose': out['smplx_lhand_pose'].detach().cpu().numpy()[0].tolist(),
                'smplx_rhand_pose': out['smplx_rhand_pose'].detach().cpu().numpy()[0].tolist(),
                'smplx_jaw_pose': out['smplx_jaw_pose'].detach().cpu().numpy()[0].tolist(),
                'smplx_shape': out['smplx_shape'].detach().cpu().numpy()[0].tolist(),
                'smplx_expr': out['smplx_expr'].detach().cpu().numpy()[0].tolist(),
                'cam_trans': out['cam_trans'].detach().cpu().numpy()[0].tolist()
            }
            results.append(frame_result)
        cap.release()
        # save all results to JSON file
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
