import os
import json
from tqdm import tqdm
import shutil

def make_folders(path="output"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]


def convert_coco_json_to_yolo_txt(output_path, json_file):
    dat = 0
    dat_1 = 0
    path = make_folders(output_path)

    with open(json_file) as f:
        json_data = json.load(f)

    # lables files holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "object.labels")
    with open(label_file, "w") as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            f.write(f"{category_name}\n")

    for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        #anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        img_temp = img_name.split("/")
        anno_txt = str(output_path)+ "/"+str(img_temp[-1].split(".")[0]) + ".txt"
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"]
                if (anno["category_id"] == 1):
                    bbox_COCO = anno["bbox"]
                    x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                    f.write(f"{0} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    #dat +=1
                    #print (img_name)
                if (anno["category_id"] == 3):
                    #dat_1 +=1
                    bbox_COCO = anno["bbox"]
                    x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                    f.write(f"{1} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Converting COCO Json to YOLO txt finished!")
    print (dat)
    print (dat_1)


convert_coco_json_to_yolo_txt("output","thermal_annotations.json")