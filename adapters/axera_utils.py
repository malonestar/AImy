# axera_utils.py
import os
import cv2
import numpy as np
import axengine as ax
from dataclasses import dataclass

# --- Dataclasses ---
@dataclass
class Object:
    bbox: list; label: int; prob: float

@dataclass
class PoseObject:
    bbox: list; label: int; prob: float; kps: np.ndarray

# --- Helper Functions ---
def sigmoid(x):
    x = np.clip(x, -50, 50) 
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True); e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def decode_distributions(feat, reg_max=16):
    prob = softmax(feat, axis=-1); dist = np.sum(prob * np.arange(reg_max), axis=-1)
    return dist

class Detector:
    def __init__(self, model_path: str, labels_path: str = "coco.txt", conf_thres: float = 0.45, iou_thres: float = 0.45):
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path): raise FileNotFoundError(f"Labels file not found: {labels_path}")
        self.sess = ax.InferenceSession(model_path, providers=["AXCLRTExecutionProvider"])
        self.conf_thres = conf_thres; self.iou_thres = iou_thres; self.reg_max = 16
        with open(labels_path, 'r') as f: self.labels = [line.strip() for line in f.readlines()]
        np.random.seed(42)
        self.colors = [tuple(np.random.randint(0, 255, size=3).tolist()) for _ in self.labels]
        input_details = self.sess.get_inputs()[0]
        self.input_name = input_details.name; self.input_shape = tuple(input_details.shape)
        self.is_chw = self.input_shape[1] == 3
        self.model_h = self.input_shape[2] if self.is_chw else self.input_shape[1]
        self.model_w = self.input_shape[3] if self.is_chw else self.input_shape[2]
        print("[INFO] Object Detector initialized successfully.")

    def _letterbox(self, frame_rgb: np.ndarray):
        original_h, original_w = frame_rgb.shape[:2]; r = min(self.model_w / original_w, self.model_h / original_h)
        new_w, new_h = int(original_w * r), int(original_h * r)
        resized_img = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.full((self.model_h, self.model_w, 3), 114, dtype=np.uint8)
        dw, dh = (self.model_w - new_w) // 2, (self.model_h - new_h) // 2
        padded_img[dh:dh + new_h, dw:dw + new_w] = resized_img
        return padded_img, r, dw, dh

    def infer_single(self, preprocessed_frame, original_shape, ratio, pad_w, pad_h):
        input_tensor = np.expand_dims(preprocessed_frame, axis=0)
        if self.is_chw: input_tensor = input_tensor.transpose(0, 3, 1, 2)
        outputs = self.sess.run(None, {self.input_name: input_tensor})
        return self._postprocess(outputs, original_shape, ratio, pad_w, pad_h)

    def _postprocess(self, outputs: list, original_shape: tuple, ratio: float, pad_w: int, pad_h: int):
        strides = [8, 16, 32]; detections = []; num_classes = len(self.labels)
        bbox_channels = 4 * self.reg_max
        for i, stride in enumerate(strides):
            output = outputs[i]; batch_size, grid_h, grid_w, channels = output.shape
            bbox_part = output[..., :bbox_channels].reshape(batch_size, grid_h * grid_w, 4, self.reg_max)
            class_part = output[..., bbox_channels:].reshape(batch_size, grid_h * grid_w, num_classes)
            for b in range(batch_size):
                class_scores = sigmoid(class_part[b]); box_prob = class_scores.max(axis=-1)
                valid_indices = box_prob > self.conf_thres
                if not np.any(valid_indices): continue
                class_ids = class_scores.argmax(axis=-1)[valid_indices]
                box_prob = box_prob[valid_indices]; bbox_decode = bbox_part[b, valid_indices, :, :]
                left=decode_distributions(bbox_decode[:,0,:],self.reg_max); top=decode_distributions(bbox_decode[:,1,:],self.reg_max)
                right=decode_distributions(bbox_decode[:,2,:],self.reg_max); bottom=decode_distributions(bbox_decode[:,3,:],self.reg_max)
                grid_indices = np.where(valid_indices)[0]
                h_coords = grid_indices // grid_w; w_coords = grid_indices % grid_w
                pb_cx = (w_coords + 0.5) * stride; pb_cy = (h_coords + 0.5) * stride
                x0=(pb_cx-left*stride-pad_w)/ratio; y0=(pb_cy-top*stride-pad_h)/ratio
                x1=(pb_cx+right*stride-pad_w)/ratio; y1=(pb_cy+bottom*stride-pad_h)/ratio
                x0=np.clip(x0,0,original_shape[1]); y0=np.clip(y0,0,original_shape[0])
                x1=np.clip(x1,0,original_shape[1]); y1=np.clip(y1,0,original_shape[0])
                for j in range(len(box_prob)):
                    detections.append(Object(bbox=[x0[j],y0[j],x1[j]-x0[j],y1[j]-y0[j]],label=class_ids[j],prob=box_prob[j]))
        if not detections: return []
        boxes=np.array([d.bbox for d in detections]); scores=np.array([d.prob for d in detections])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
        if len(indices) == 0: return []
        return [detections[i] for i in indices.flatten()]

    def draw_detections(self, frame: np.ndarray, detections: list):
        for det in detections:
            bbox, score, class_id = det.bbox, det.prob, det.label
            label_text = self.labels[class_id]
            color = self.colors[class_id]
            x, y, w, h = map(int, bbox)

            # Draw the bounding box (using the random color)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"{label_text} {score:.1%}"
            font_scale = 0.5
            font_thickness = 2
            text_color = (255, 255, 255)
            bg_color = color

            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Adjust label_y position to prevent going off-screen
            if y - label_height - baseline - 5 < 0:
                label_x = x + 2
                label_y = y + label_height + baseline + 2
                cv2.rectangle(frame, (label_x, label_y - label_height - baseline), (label_x + label_width, label_y), bg_color, -1)
                cv2.putText(frame, label, (label_x, label_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            else:
                label_x = x
                label_y = y - 5
                cv2.rectangle(frame, (label_x, label_y - label_height - baseline), (label_x + label_width, label_y), bg_color, -1)
                cv2.putText(frame, label, (label_x, label_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        return frame

class PoseDetector:
    def __init__(self, model_path: str, conf_thres: float = 0.5, iou_thres: float = 0.45):
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
        self.sess = ax.InferenceSession(model_path, providers=["AXCLRTExecutionProvider"])
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.reg_max = 16 # For DFL decoding of boxes
        
        input_details = self.sess.get_inputs()[0]
        self.input_name = input_details.name; self.input_shape = tuple(input_details.shape)
        self.is_chw = self.input_shape[1] == 3
        self.model_h = self.input_shape[2] if self.is_chw else self.input_shape[1]
        self.model_w = self.input_shape[3] if self.is_chw else self.input_shape[2]
        
        # COCO Pose Skeleton Structure
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        self.limb_colors = [[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,0],[0,255,85],[0,255,170],[0,255,255],[0,170,255],[0,85,255],[0,0,255],[85,0,255],[170,0,255],[255,0,255],[255,0,170],[255,0,85]]
        self.kpt_colors = [[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,0],[0,255,85],[0,255,170],[0,255,255],[0,170,255],[0,85,255],[0,0,255],[85,0,255],[170,0,255],[255,0,255],[255,0,170]] # 17 keypoints
        self.bbox_color = (0, 0, 255)
        print("[INFO] Pose Detector initialized successfully.")

    def _letterbox(self, frame_rgb: np.ndarray):
        original_h, original_w = frame_rgb.shape[:2]; r = min(self.model_w / original_w, self.model_h / original_h)
        new_w, new_h = int(original_w * r), int(original_h * r)
        resized_img = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.full((self.model_h, self.model_w, 3), 114, dtype=np.uint8)
        dw, dh = (self.model_w - new_w) // 2, (self.model_h - new_h) // 2
        padded_img[dh:dh + new_h, dw:dw + new_w] = resized_img
        return padded_img, r, dw, dh
    
    def _postprocess_pose(self, outputs: list, ratio: float, pad_w: int, pad_h: int):
        det_outputs = outputs[:3]; kps_outputs = outputs[3:6]
        strides = [8, 16, 32]; all_candidates = []
        for i, stride in enumerate(strides):
            det_head = np.squeeze(det_outputs[i]); kps_head = np.squeeze(kps_outputs[i])
            if det_head.ndim != 3: continue
            grid_h, grid_w, det_channels = det_head.shape
            flat_dets = det_head.reshape(grid_h * grid_w, det_channels)
            flat_kps = kps_head.reshape(grid_h * grid_w, -1)
            box_dist = flat_dets[:, :64]; scores = sigmoid(flat_dets[:, 64])
            valid_indices_mask = scores > self.conf_thres
            if not np.any(valid_indices_mask): continue
            box_dist = box_dist[valid_indices_mask]; scores = scores[valid_indices_mask]
            kps_data = flat_kps[valid_indices_mask]
            box_dist = box_dist.reshape(-1, 4, self.reg_max)
            lt = decode_distributions(box_dist[:, :2, :]); rb = decode_distributions(box_dist[:, 2:, :])
            grid_indices = np.where(valid_indices_mask)[0]
            grid_y = grid_indices // grid_w; grid_x = grid_indices % grid_w
            xy1 = (np.column_stack([grid_x, grid_y]) - lt) * stride
            xy2 = (np.column_stack([grid_x, grid_y]) + rb) * stride
            boxes = np.concatenate((xy1, xy2 - xy1), axis=1) # [x1, y1, w, h]
            kps = kps_data.reshape(-1, 17, 3)
            kps[..., 0] = (kps[..., 0] * 2 + grid_x[:, None]) * stride
            kps[..., 1] = (kps[..., 1] * 2 + grid_y[:, None]) * stride
            kps[..., 2] = sigmoid(kps[..., 2])
            for box, score, kpt_set in zip(boxes, scores, kps):
                all_candidates.append({'box': box, 'score': score, 'kps': kpt_set})
        if not all_candidates: return []
        boxes_for_nms = [cand['box'] for cand in all_candidates]
        scores_for_nms = [cand['score'] for cand in all_candidates]
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, self.conf_thres, self.iou_thres)
        if len(indices) == 0: return []
        final_poses = []
        for i in indices.flatten():
            cand = all_candidates[i]; box = cand['box']; kps = cand['kps']
            box[0] = (box[0] - pad_w) / ratio; box[1] = (box[1] - pad_h) / ratio
            box[2] /= ratio; box[3] /= ratio
            kps[..., 0] = (kps[..., 0] - pad_w) / ratio
            kps[..., 1] = (kps[..., 1] - pad_h) / ratio
            final_poses.append(PoseObject(bbox=box, label=0, prob=cand['score'], kps=kps))
        return final_poses

    def infer_single(self, preprocessed_frame, original_shape, ratio, pad_w, pad_h):
        input_tensor = np.expand_dims(preprocessed_frame, axis=0)
        if self.is_chw: input_tensor = input_tensor.transpose(0, 3, 1, 2)
        outputs = self.sess.run(None, {self.input_name: input_tensor})
        num_outputs = len(outputs)
        if num_outputs != 6: # Basic check for expected output structure
             print(f"[WARN] Expected 6 outputs (3 det + 3 kps), but got {num_outputs}. Postprocessing might fail.")
             if num_outputs < 4: return []
        return self._postprocess_pose(outputs, ratio, pad_w, pad_h)

    def draw_poses(self, frame: np.ndarray, poses: list):
        if not poses: return frame
        for pose in poses:
            x, y, w, h = map(int, pose.bbox)
            # Use blue color for box as in the example
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.bbox_color, 1) # Thin box
            label = f"person {pose.prob:.1%}"; font_scale = 0.5; font_thickness = 2
            text_color = (0, 0, 0); bg_color = (255, 255, 255)
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_x = x; label_y = max(y - 5, label_height + baseline + 5)
            cv2.rectangle(frame, (label_x, label_y - label_height - baseline), (label_x + label_width, label_y), bg_color, -1)
            cv2.putText(frame, label, (label_x, label_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            kps = pose.kps
            for i, limb in enumerate(self.skeleton):
                p1_idx, p2_idx = limb[0] - 1, limb[1] - 1
                kpt_conf_threshold = 0.5
                if kps[p1_idx, 2] > kpt_conf_threshold and kps[p2_idx, 2] > kpt_conf_threshold:
                    p1 = (int(kps[p1_idx, 0]), int(kps[p1_idx, 1]))
                    p2 = (int(kps[p2_idx, 0]), int(kps[p2_idx, 1]))
                    cv2.line(frame, p1, p2, self.limb_colors[i % len(self.limb_colors)], 2)
            for j in range(len(kps)):
                if kps[j, 2] > kpt_conf_threshold:
                    pt = (int(kps[j, 0]), int(kps[j, 1]))
                    cv2.circle(frame, pt, 4, self.kpt_colors[j % len(self.kpt_colors)], -1)
        return frame