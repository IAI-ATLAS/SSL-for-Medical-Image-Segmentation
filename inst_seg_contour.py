import numpy as np
import cv2
import random
import colorsys
from skimage.measure import label as label_fcn
from glob import glob


def random_colors(N, bright=True):
    """Generate random colors.
    
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    random.seed(10)
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    #return len(colors)*[(0,1,1)]
    #return len(colors)*[(0,1,0)]
    return colors


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def visualize_instances_map(
    input_image, inst_map, type_map=None, type_colour=None, line_thickness=1
):
    """Overlays segmentation results on image as contours.
    Args:
        input_image: input image
        inst_map: instance mask with unique value for every object
        type_map: type mask with unique value for every class
        type_colour: a dict of {type : colour} , `type` is from 0-N
                     and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    Returns:
        overlay: output image with segmentation overlay as contours
    """
    overlay = np.copy((input_image).astype(np.uint8))

    inst_list = list(np.unique(inst_map))  # get list of instances
    inst_list.remove(0)  # remove background

    inst_rng_colors = random_colors(len(inst_list))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for inst_idx, inst_id in enumerate(inst_list):
        inst_map_mask = np.array(inst_map == inst_id, np.uint8)  # get single object
        y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst_map_crop = inst_map_mask[y1:y2, x1:x2]
        contours_crop = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # only has 1 instance per map, no need to check #contour detected by opencv
        contours_crop = np.squeeze(
            contours_crop[0][0].astype("int32")
        )  # * opencv protocol format may break
        if contours_crop.ndim==1:
            contours_crop = np.expand_dims(contours_crop,0)
        contours_crop += np.asarray([[x1, y1]])  # index correction
        if type_map is not None:
            type_map_crop = type_map[y1:y2, x1:x2]
            type_id = np.unique(type_map_crop).max()  # non-zero
            inst_colour = type_colour[type_id]
        else:
            inst_colour = (inst_rng_colors[inst_idx]).tolist()
           # inst_colour = type_colour[inst_id]
        cv2.drawContours(overlay, [contours_crop], -1, inst_colour, line_thickness)
    return overlay



dirs = glob("/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/monuseg_cam/detco/")

img_0 = cv2.cvtColor(cv2.imread('/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/monuseg_cam/10_cropped.png',1), cv2.COLOR_BGRA2BGR)
# mask = cv2.cvtColor(cv2.imread('/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/monuseg_cam/10_cropped_mask.png',1), cv2.COLOR_BGR2GRAY)
# overlay = visualize_instances_map(img_0,label_fcn(mask>0*255),line_thickness=2)
# cv2.imwrite('/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/monuseg_cam/10_cropped_mask_overlay.png',overlay)

img_72 = cv2.cvtColor(cv2.imread('/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/monuseg_cam/11_cropped.png',1), cv2.COLOR_BGRA2BGR)
# mask = cv2.cvtColor(cv2.imread('/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/monuseg_cam/11_cropped_mask.png',1), cv2.COLOR_BGR2GRAY)
# overlay = visualize_instances_map(img_72,label_fcn(mask>0*255),line_thickness=2)
# cv2.imwrite('/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/monuseg_cam/11_cropped_mask_overlay.png',overlay)


img_75 = cv2.cvtColor(cv2.imread('/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/monuseg_cam/13_cropped.png',1), cv2.COLOR_BGRA2BGR)
# mask = cv2.cvtColor(cv2.imread('/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/monuseg_cam/13_cropped_mask.png',1), cv2.COLOR_BGR2GRAY)
# overlay = visualize_instances_map(img_75,label_fcn(mask>0*255),line_thickness=2)
# cv2.imwrite('/home/ws/kg2371/projects/self-supervised-biomedical-image-segmentation/monuseg_cam/13_cropped_mask_overlay.png',overlay)



for dir in dirs:

    for subdir in [f'{dir}EigenCAM', f'{dir}FullGrad', f'{dir}XGradCAM']:

        mask = cv2.cvtColor(cv2.imread(f'{subdir}/10_cropped_pred_mask.png',1), cv2.COLOR_BGR2GRAY)
        overlay = visualize_instances_map(img_0,label_fcn(mask>0*255),line_thickness=2)
        cv2.imwrite(f'{subdir}/10_cropped_pred_mask_overlay.png',overlay)

        mask = cv2.cvtColor(cv2.imread(f'{subdir}/11_cropped_pred_mask.png',1), cv2.COLOR_BGR2GRAY)
        overlay = visualize_instances_map(img_72,label_fcn(mask>0*255),line_thickness=2)
        cv2.imwrite(f'{subdir}/11_cropped_pred_mask_overlay.png',overlay)

        mask = cv2.cvtColor(cv2.imread(f'{subdir}/13_cropped_pred_mask.png',1), cv2.COLOR_BGR2GRAY)
        overlay = visualize_instances_map(img_75,label_fcn(mask>0*255),line_thickness=2)
        cv2.imwrite(f'{subdir}/13_cropped_pred_mask_overlay.png',overlay)


# imname = 'gt'

# img = cv2.cvtColor(cv2.imread('/home/ws/kg2371/Promotion/04_2022/bmt_2022/5/x.png',1), cv2.COLOR_BGRA2BGR)
# mask = cv2.cvtColor( cv2.imread(f'/home/ws/kg2371/Promotion/04_2022/bmt_2022/5/{imname}.png',1), cv2.COLOR_BGR2GRAY)
# overlay = visualize_instances_map(img,label_fcn(mask>0*255),type_colour={255:(255,250,205)},line_thickness=1)
# cv2.imwrite(f'/home/ws/kg2371/Promotion/04_2022/bmt_2022/5/{imname}_overlay.png',overlay)