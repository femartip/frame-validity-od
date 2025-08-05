def pq(detections, annotations: list) -> float:
    #https://arxiv.org/pdf/1801.00868
    iou_tp, tp, fp, fn = match_detections_to_annotations(detections, annotations)
    pq = sum(iou_tp)/(tp + 0.5*fp + 0.5*fn)
    return pq

def lrp(detections, annotations: list, tau: float = 0.5) -> float:
    #https://arxiv.org/pdf/1807.01696
    iou_tp, tp, fp, fn = match_detections_to_annotations(detections, annotations)
    lrp = (sum([iou/tau for iou in iou_tp]) + fp + fn)/(tp + fp + fn)
    return lrp

def calculate_iou(box1: list[int], box2: list[int]) -> float:
    def center_to_corner(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2  
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    
    box1_corner = center_to_corner(box1)
    box2_corner = center_to_corner(box2)
    
    x1 = max(box1_corner[0], box2_corner[0])
    y1 = max(box1_corner[1], box2_corner[1])
    x2 = min(box1_corner[2], box2_corner[2])
    y2 = min(box1_corner[3], box2_corner[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    iou = ((x2 - x1) * (y2 - y1))/((box1[2] * box1[3]) + (box2[2] * box2[3]) - ((x2 - x1) * (y2 - y1)))
    return iou

def match_detections_to_annotations(detections, annotations: list) -> tuple[list, int, int, int]:
    iou = []
    used_annotations = set()
    
    for i, (det_box, det_class, det_conf) in enumerate(zip(detections['coordinates'], detections['class'], detections['confidence'])):
        best_iou = 0
        best_match = -1
        
        for j, (ann_class, ann_x, ann_y, ann_w, ann_h) in enumerate(annotations):
            if j in used_annotations:
                continue
            if int(det_class) == ann_class:
                inst_iou = calculate_iou(det_box, [ann_x, ann_y, ann_w, ann_h])
                if inst_iou > best_iou and inst_iou:
                    best_iou = inst_iou
                    best_match = j
        
        if best_match != -1:
            used_annotations.add(best_match)
            iou.append(best_iou)
        
    tp = len(iou)
    fp = len(detections['coordinates']) - len(iou)
    fn = len(annotations) - len(iou)
    return iou, tp, fp, fn

if __name__ == "__main__":
    detections = {
        'coordinates': [[1,1,2,2], [10,10,2,2]],
        'class':      [1,       1],
        'confidence': [0.9,     0.8]
    }
    annotations = [
        (1, 1, 1, 2, 2),
        (1, 5, 5, 2, 2)
    ]

    pq_val = pq(detections, annotations)
    lrp_val = lrp(detections, annotations, tau=0.5)

    assert abs(pq_val - 0.5) < 1e-6
    assert abs(lrp_val - (4/3)) < 1e-6

    print(pq_val)
    print(lrp_val)
