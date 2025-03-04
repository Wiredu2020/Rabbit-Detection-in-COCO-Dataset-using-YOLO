import torch
import utils.config as config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image_with_boxes(image, target):
    """
    Plot an image with its bounding box from the YOLO format target.
    """
    image = image.permute(1, 2, 0).numpy()
    
    # Unnormalize the image
    image = image * 0.25 + 0.5
    img_size = image.shape[0]
    
    # Find cells with objects
    obj_mask = target[..., 4] > 0.1
    
    # Get the indices of cells with objects
    obj_indices = torch.nonzero(obj_mask)[0]
    
    # Take first object cell
    i, j = obj_indices
    cell = target[i, j]
    confidence = cell[4].item()
    
    # construct box_coords with grid cell location and x, y, w, h
    box_coords = torch.tensor([j, i,*cell[:4]])
    
    # Convert to bbox
    bbox = convert_yolo_to_bbox(box_coords, img_size)
    
    _, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image)
    draw_bounding_box(ax, bbox, confidence)
    
    plt.axis('off')
    plt.show()
    
def draw_bounding_box(ax, box, confidence, color="red", alpha=1):
    """
    Draws given bounding boxes to the given axis, and adds a label to the bounding box
    """
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        facecolor="none",
        edgecolor=color,
        linewidth=2,
        alpha=alpha,
    )
    ax.add_patch(rect)
    if confidence is not None:
        ax.text(
            box[0], box[1] - 5,
            f'Rabbit: {confidence:.2f}',
            color='white', fontsize=10,
            bbox=dict(facecolor='red', alpha=0.6)
        )


def convert_yolo_to_bbox(yolo_box, img_size, grid_size=config.GRID_SIZE):
    """
    Convert YOLO format (x_center, y_center, width, height) relative to grid cell
    to absolute bbox coordinates (x_min, y_min, x_max, y_max)
    """
    # Extract coordinates
    grid_x, grid_y = yolo_box[0], yolo_box[1]  # Grid cell index
    x_rel, y_rel = yolo_box[2], yolo_box[3]    # x, y relative to grid cell
    w_rel, h_rel = yolo_box[4], yolo_box[5]    # w, h relative to image

    # Convert to absolute coordinates
    cell_size = img_size / grid_size
    x_center = (grid_x + x_rel) * cell_size
    y_center = (grid_y + y_rel) * cell_size
    width = w_rel * img_size  # Absolute width
    height = h_rel * img_size  # Absolute height

    # Convert to top-left (x_min, y_min) and bottom-right (x_max, y_max)
    x_min = np.clip(x_center - width / 2, 0, img_size - 1)
    y_min = np.clip(y_center - height / 2, 0, img_size - 1)
    x_max = np.clip(x_center + width / 2, 0, img_size - 1)
    y_max = np.clip(y_center + height / 2, 0, img_size - 1)

    return np.array([x_min, y_min, x_max, y_max])


def find_best_bbox(boxes, image_size):
    """
    Returns the most confident bounding box
    """
    grid_size = config.GRID_SIZE
    
    # get most confident prediction
    confidences = boxes[..., 4]
    best_loc = torch.argmax(confidences)  # Flattened index
    
    # Convert to grid coordinates row (i), column (j)
    i, j = divmod(best_loc.item(), grid_size)

    # Extract (x, y, w, h) for the best cell
    best = boxes[i, j, :4]  # (x, y, w, h)
    confidence = confidences[i, j].clone()  # Extract confidence

    best = torch.cat([torch.tensor([j, i]), best])
    # Convert from YOLO format (normalized relative) to absolute coordinates
    bbox = convert_yolo_to_bbox(best, image_size, grid_size)
    
    return bbox, confidence

def compute_iou(bbox1, bbox2): 
    """
    Compute IoU between box1 and box2
    box format: [x1, y1, x2, y2]
    """
    def sort_coords(x1, y1, x2, y2):
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    x11, y11, x12, y12 = sort_coords(*bbox1)
    x21, y21, x22, y22 = sort_coords(*bbox2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)
    
    intersection_area = max(0, (x_right - x_left) * (y_bottom - y_top))
    bbox1_area = (x12 - x11) * (y12 - y11)
    bbox2_area = (x22 - x21) * (y22 - y21)
    union_area = bbox1_area + bbox2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0
