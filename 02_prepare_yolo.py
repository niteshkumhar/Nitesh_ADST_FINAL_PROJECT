import os
import shutil
from pathlib import Path

CLASS_MAP = {
    1: 0,  # Open
    2: 1,  # Short
    3: 2,  # Mousebite
    4: 3,  # Spur
    5: 4,  # Copper
    6: 5   # Pin-hole
}

IMG_WIDTH = 640.0
IMG_HEIGHT = 640.0

def convert_bbox_to_yolo(x1, y1, x2, y2):
    dw = 1.0 / IMG_WIDTH
    dh = 1.0 / IMG_HEIGHT
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x_center * dw, y_center * dh, w * dw, h * dh

def prepare_yolo_dataset():
    base_dir = "DeepPCB-master"
    pcb_data_dir = os.path.join(base_dir, "PCBData")
    
    train_txt_path = os.path.join(pcb_data_dir, "my_trainval.txt")

    output_base = "yolo_dataset"
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_base, "labels", split), exist_ok=True)

    if not os.path.exists(train_txt_path):
        print(f"Error: Could not find '{train_txt_path}'!")
        return

    print(f"Processing dataset from {train_txt_path}...")
    with open(train_txt_path, "r") as f:
        lines = f.readlines()

    total_converted = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        rel_img_path = parts[0]    # e.g., group00041/00041/00041000_temp.jpg

        src_img = os.path.join(pcb_data_dir, rel_img_path.replace('/', os.sep))
        if not os.path.exists(src_img):
            continue

        # Clean base name to match annotation files (e.g. 00041000)
        img_file_name = Path(src_img).name
        sample_stem = Path(src_img).stem
        clean_stem = sample_stem.replace("_temp", "").replace("_test", "").replace("_template", "")

        # Reconstruct the correct annotation path (without suffixes in the txt filename)
        # rel_img_path looks like: group00041/00041/00041000_temp.jpg
        # We want: group00041/00041_not/00041000.txt
        path_parts = rel_img_path.split('/')
        if len(path_parts) >= 3:
            group_folder = path_parts[0]
            not_folder_name = f"{path_parts[1]}_not"
            src_annot = os.path.join(pcb_data_dir, group_folder, not_folder_name, f"{clean_stem}.txt")
        else:
            continue

        if not os.path.exists(src_annot):
            continue

        # 80% train, 20% validation split
        split_name = "val" if total_converted % 5 == 0 else "train"

        dest_img = os.path.join(output_base, "images", split_name, f"{clean_stem}.jpg")
        dest_label = os.path.join(output_base, "labels", split_name, f"{clean_stem}.txt")

        # Copy image file to YOLO structure
        shutil.copyfile(src_img, dest_img)

        # Convert annotations to YOLO format
        yolo_annotations = []
        with open(src_annot, "r") as af:
            for a_line in af.readlines():
                a_parts = a_line.strip().split()
                if len(a_parts) >= 5:
                    x1, y1, x2, y2, defect_type = map(int, a_parts[:5])
                    if defect_type in CLASS_MAP:
                        yolo_cls = CLASS_MAP[defect_type]
                        xc, yc, w, h = convert_bbox_to_yolo(x1, y1, x2, y2)
                        yolo_annotations.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        with open(dest_label, "w") as out_f:
            out_f.writelines(yolo_annotations)

        total_converted += 1

    print(f"\nDataset conversion complete!")
    print(f"Total successfully processed images: {total_converted}")

    # Generate data.yaml configuration file for YOLO
    abs_output_base = os.path.abspath(output_base).replace("\\", "/")
    yaml_content = f"""path: {abs_output_base}
train: images/train
val: images/val

nc: 6
names: ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
"""
    yaml_path = os.path.join(output_base, "data.yaml")
    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)

    print(f"Created config: {yaml_path}")

if __name__ == "__main__":
    prepare_yolo_dataset()