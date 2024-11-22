import numpy as np
import supervision as sv
from copy import deepcopy
import cv2
from pycocotools import mask as maskUtils
from PIL import Image, ImageDraw

def xywh2xyxy(xywh):
    x,y,w,h = xywh
    return [x, y, x+w, y+h]

def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, np.ndarray):
        return mask_ann
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list): # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else: # rle
        rle = mask_ann
    mask = maskUtils.decode(rle).astype(bool)
    return mask

class SVVisualizer:
    def __init__(self, 
                 bbox_thickness: int = 2, 
                 mask_opacity: float = 0.5,
                 label_text_scale: float = 0.5,
                 label_text_thickness: int = 1,
                 label_text_padding: int = 10,
                 label_text_position: str = "top_left", # ["top_left", "center_of_mass"]
                 top_padding: int = 0,
                 right_padding: int = 0,
    ):
        self.bbox_annotator = sv.BoxAnnotator(thickness=bbox_thickness, 
                                                      color_lookup=sv.ColorLookup.CLASS)
        self.mask_annotator = sv.MaskAnnotator(opacity=mask_opacity,
                                               color_lookup=sv.ColorLookup.CLASS
                                               )
        self.polygon_annotator = sv.PolygonAnnotator(thickness=bbox_thickness, color_lookup=sv.ColorLookup.CLASS)
        # convert label_text_position to sv.Position
        label_text_position = getattr(sv.Position, label_text_position.upper())
        self.label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.CLASS,
            text_scale=label_text_scale,
            text_thickness=label_text_thickness,
            text_padding=label_text_padding,
            text_position=label_text_position,
        )
        
        self.top_padding = top_padding
        self.right_padding = right_padding
        
    def draw_masks(self, 
                   image_rgb: np.ndarray, 
                   detections: sv.Detections, 
                   draw_bbox: bool = True, 
                   draw_polygon: bool = False,
                   draw_mask: bool = True, 
                   draw_label: bool = True,
                   reverse_order=False,
    ) -> np.ndarray:
        image_display = image_rgb.copy() # cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        detections = deepcopy(detections)
        detections.class_id = np.arange(len(detections))
        labels = [str(i) for i in range(len(detections))]
        if reverse_order:
            detections = detections[::-1]
            labels = labels[::-1]
            
        if draw_bbox:
            image_display = self.bbox_annotator.annotate(image_display, detections)
        if draw_mask:
            image_display = self.mask_annotator.annotate(image_display, detections)
        if draw_polygon:
            image_display = self.polygon_annotator.annotate(image_display, detections)
        if draw_label: # draw label on top of image, optionally with padding
            has_padding = self.top_padding or self.right_padding
            if self.top_padding: 
                # need to shift detections down
                detections = deepcopy(detections)
                detections.xyxy[:,1] += self.top_padding
            if has_padding:
                h,w = image_rgb.shape[:2]
                # paste image_rgb onto blank_image to [0, top_padding] position
                blank_image = Image.new('RGB', (w+self.right_padding, h+self.top_padding), (255, 255, 255))
                blank_image.paste(Image.fromarray(image_display), (0, self.top_padding))
                image_display = np.array(blank_image)
                
            image_display = self.label_annotator.annotate(image_display, detections, labels)
        
        return image_display
    
    def insert_white_padding(self, image_rgb: np.ndarray) -> np.ndarray:
        h,w = image_rgb.shape[:2]
        image_with_padding = np.ones((h+self.top_padding, w+self.right_padding, 3), dtype=np.uint8) * 255
        image_with_padding[:h, :w] = image_rgb
        return image_with_padding
        
def detections_from_sam(regions: list[dict], include_mask=True) -> tuple[list, list]:
    boxes =  np.array([xywh2xyxy(region['bbox']) for region in regions])
    segmentations = None

    if len(boxes) == 0:
        return sv.Detections.empty()
    
    if include_mask:
        segmentations = np.array([annToMask(region['segmentation']) for region in regions])
    return sv.Detections(xyxy=boxes, mask=segmentations)

def visualize_masks(image_rgb: np.ndarray, regions: list[dict] | sv.Detections, 
                    draw_bbox: bool = True, draw_mask: bool = True, draw_polygon: bool = False, draw_label: bool = True,
                    label_text_position: str = "top_left",
                    white_padding: int = 0,
                    reverse_order: bool = False,
                    plot_image: bool = True) -> np.ndarray:   
    ''' Returns rgb image with masks drawn on it '''
    if isinstance(regions, sv.Detections):
        detections = regions
    else:
        detections: sv.Detections = detections_from_sam(regions, include_mask=draw_mask)
    image_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h,w = image_rgb.shape[:2]
    largest_edge = max(h,w)
    visualizer = SVVisualizer(
        label_text_padding=largest_edge//100,
        label_text_thickness=largest_edge//500,
        label_text_scale=max(0.5, largest_edge//1000),
        label_text_position=label_text_position,
        top_padding=white_padding,
        right_padding=white_padding,
    )
    image_display: np.ndarray = visualizer.draw_masks(image_display, detections,
                                                      draw_bbox=draw_bbox, 
                                                      draw_mask=draw_mask, 
                                                      draw_polygon=draw_polygon,
                                                      draw_label=draw_label,
                                                      reverse_order=reverse_order
                                                      )  
    image_with_masks = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)

    if plot_image:
        sv.plot_image(image_display)
    
    return image_with_masks