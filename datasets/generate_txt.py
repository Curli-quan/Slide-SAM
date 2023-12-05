import os
import glob
import shutil
from tutils import tfilename


HOME_PATH="/quanquan/datasets/"

def check_existing(img_path, label_path):
    if os.path.exists(img_path) and os.path.exists(label_path):
        return True
    else: 
        if not os.path.exists(img_path):
            print("IMAGE Not exist: ", img_path)
        if not os.path.exists(label_path):
            print("LABEL Not exist: ", label_path)
        return False

def get_availabel_files(names, label_names):
    llist = [[n,n2] for n,n2 in zip(names, label_names) if check_existing(n, n2)]
    names = [n[0] for n in llist]
    label_names = [n[1] for n in llist]
    return names, label_names
    
def write_txt(img_paths, label_paths, meta_info, split, writing_mode='a+'):
    dataset_id = meta_info['dataset_id']
    assert split in ['train', 'val', 'test'], f" split in ['train', 'val', 'test'] , but Got {split}"
    save_path = meta_info["save_txt_path"].replace("_train.txt", f"_{split}.txt")
    
    count = 0
    with open(save_path, writing_mode) as f:
        for p1, p2 in zip(img_paths, label_paths):
            p1 = p1.replace(meta_info['home_path'], "")
            p2 = p2.replace(meta_info['home_path'], "")
            line = f"{p1}\t{p2}\n"
            # print(line, end=" ")
            f.write(line)
            count += 1
    if count <= 0:
        raise ValueError(f"ID: {meta_info['dataset_id']}, \tTask: {meta_info['dataset_name']}\t, {count} files are writen.")
    print(f"ID: {meta_info['dataset_id']}, \tTask: {meta_info['dataset_name']}\t, {count} files are writen.\t Writing Over! write into ", save_path)


def organize_in_nnunet_style(meta_info):
    dirpath = os.path.join(meta_info['home_path'], meta_info['dirpath'])
    if os.path.exists(os.path.join(dirpath, "imagesTr")) and os.path.exists(os.path.join(dirpath, "labelsTr")):
        img_paths = glob.glob(os.path.join(dirpath, "imagesTr", "*.nii.gz"))
        img_paths.sort()
        label_paths = [p.replace("imagesTr", "labelsTr")[:-12]+".nii.gz" for p in img_paths]
        img_paths, label_paths = get_availabel_files(img_paths, label_paths)
        write_txt(img_paths, label_paths, meta_info=meta_info, split='train', writing_mode="a+")

    if os.path.exists(os.path.join(dirpath, "imagesVa")) and os.path.exists(os.path.join(dirpath, "labelsVa")):
        img_paths = glob.glob(os.path.join(dirpath, "imagesVa", "*.nii.gz"))
        img_paths.sort()
        label_paths = [p.replace("imagesVa", "labelsVa")[:-12]+".nii.gz" for p in img_paths]
        img_paths, label_paths = get_availabel_files(img_paths, label_paths)
        write_txt(img_paths, label_paths, meta_info=meta_info, split='val', writing_mode="a+")
        
    if os.path.exists(os.path.join(dirpath, "imagesTs")) and os.path.exists(os.path.join(dirpath, "labelsTs")):
        img_paths = glob.glob(os.path.join(dirpath, "imagesTs", "*.nii.gz"))
        img_paths.sort()
        label_paths = [p.replace("imagesTs", "labelsTs")[:-12]+".nii.gz" for p in img_paths]
        img_paths, label_paths = get_availabel_files(img_paths, label_paths)
        write_txt(img_paths, label_paths, meta_info=meta_info, split='test', writing_mode="a+")


def organize_in_style2(meta_info):    
    dirpath = os.path.join(meta_info['home_path'], meta_info['dirpath'])
    if os.path.exists(os.path.join(dirpath, "imagesTr")) and os.path.exists(os.path.join(dirpath, "labelsTr")):
        img_paths = glob.glob(os.path.join(dirpath, "imagesTr", "*.nii.gz"))
        img_paths.sort()
        label_paths = [p.replace("imagesTr", "labelsTr") for p in img_paths]
        img_paths, label_paths = get_availabel_files(img_paths, label_paths)
        write_txt(img_paths, label_paths, meta_info=meta_info, split='train', writing_mode="a+")

    if os.path.exists(os.path.join(dirpath, "imagesVa")) and os.path.exists(os.path.join(dirpath, "labelsVa")):
        img_paths = glob.glob(os.path.join(dirpath, "imagesVa", "*.nii.gz"))
        img_paths.sort()
        label_paths = [p.replace("imagesVa", "labelsVa") for p in img_paths]
        img_paths, label_paths = get_availabel_files(img_paths, label_paths)
        write_txt(img_paths, label_paths, meta_info=meta_info, split='val', writing_mode="a+")
        
    if os.path.exists(os.path.join(dirpath, "imagesTs")) and os.path.exists(os.path.join(dirpath, "labelsTs")):
        img_paths = glob.glob(os.path.join(dirpath, "imagesTs", "*.nii.gz"))
        img_paths.sort()
        label_paths = [p.replace("imagesTs", "labelsTs") for p in img_paths]
        img_paths, label_paths = get_availabel_files(img_paths, label_paths)
        write_txt(img_paths, label_paths, meta_info=meta_info, split='test', writing_mode="a+")


def organize_by_names(names_in, label_names_in, meta_info):    
    assert len(names_in) > 0, f"Meta info: {meta_info}"
    names, label_names = get_availabel_files(names_in, label_names_in)
    assert len(names) > 0, f"Meta info: {meta_info}, \n {names_in[:2]} \n {label_names_in[:2]}"
    assert len(label_names) > 0, f"Meta info: {meta_info}, \n {names_in[:2]} \n {label_names_in[:2]}"

    # print("debug files", len(names))
    if len(names) > 10:
        num_valid = min(int(len(names) // 10), 10)
        # print("num valid", num_valid)
        train_names = names[:-num_valid*2]
        valid_names = names[-num_valid*2:-num_valid]
        test_names = names[-num_valid:]

        train_labels = label_names[:-num_valid*2]
        valid_labels = label_names[-num_valid*2:-num_valid]
        test_labels = label_names[-num_valid:]

        write_txt(train_names, train_labels, meta_info=meta_info, split="train")
        write_txt(valid_names, valid_labels, meta_info=meta_info, split="val")
        write_txt(test_names, test_labels, meta_info=meta_info, split="test")
    else:
        write_txt(names, label_names, meta_info=meta_info, split="train")


def clear_files(train_path):
    if os.path.exists(train_path):
        parent, name = os.path.split(train_path)
        shutil.move(train_path, tfilename(parent, "misc", name))
    
    val_path = train_path.replace("_train.txt",  "_val.txt")
    if os.path.exists(val_path):
        parent, name = os.path.split(val_path)
        shutil.move(val_path, os.path.join(parent, "misc", name))
    
    test_path = train_path.replace("_train.txt", "_test.txt")
    if os.path.exists(test_path):
        parent, name = os.path.split(test_path)
        shutil.move(test_path, os.path.join(parent, "misc", name))
    print("Files cleared!")

# from tutils.nn.data import read
# def convert_to_nii(paths):
    

###################################################################################

###################################################################################

def get_BCV_Abdomen(save_path=None):
    meta_info = {
        "dataset_name": "BTCV",
        "dataset_id": "01",
        "modality": "CT",
        "home_path": HOME_PATH,
        "dirpath": "01_BCV-Abdomen/Training/",
        "save_txt_path": save_path,
    }
    names = glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "img/*.nii.gz"))
    names.sort()
    label_names = [p.replace("img", "label") for p in names]
    organize_by_names(names, label_names, meta_info=meta_info)

def get_AbdomenCT_1K(save_path):
    meta_info = {
        "dataset_name": "AbdomenCT-1K",
        "dataset_id": "08",
        "modality": "CT",
        "home_path": HOME_PATH,
        "dirpath": "08_AbdomenCT-1K",
        "save_txt_path": save_path,
    }
    # print(names)
    organize_in_nnunet_style(meta_info=meta_info)

def get_AMOS(save_path):
    meta_info = {
        "dataset_name": "AMOS",
        "dataset_id": "09",
        "modality": "CT",
        "home_path": HOME_PATH,
        "dirpath": "09_AMOS",
        "save_txt_path": save_path,
    }
    organize_in_style2(meta_info)

def get_MSD(save_path):
    meta_info = {
        "dataset_name": "MSD", # Decathlon
        "dataset_id": "10",
        "modality": "CT",
        "home_path": HOME_PATH,
        "parent_dirpath": "10_Decathlon",
        "dirpath": "",
        "save_txt_path": save_path,
    }
    subtasks = ["Task06_Lung", "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon"]
    for task in subtasks:
        # print("Processing ", task)
        meta_info_subtask = {
            "dataset_name":task, 
            "dataset_id": f"{meta_info['dataset_id']}_{task[4:6]}", 
            "home_path":HOME_PATH,
            "dirpath": f"{meta_info['parent_dirpath']}/{task}",
            "save_txt_path": save_path,
            }
        # print(meta_info_subtask)
        organize_in_style2(meta_info=meta_info_subtask)


def get_MSD_MRI(save_path):
    meta_info = {
        "dataset_name": "MSD", # Decathlon
        "dataset_id": "10",
        "modality": "MRI",
        "home_path": HOME_PATH,
        "parent_dirpath": "10_Decathlon",
        "dirpath": "",
        "save_txt_path": save_path,
    }
    subtasks = ["Task02_Heart", "Task05_Prostate"]
    for task in subtasks:
        # print("Processing ", task)
        meta_info_subtask = {
            "dataset_name":task, 
            "dataset_id": f"{meta_info['dataset_id']}_{task[4:6]}", 
            "home_path":HOME_PATH,
            "dirpath": f"{meta_info['parent_dirpath']}/{task}",
            "save_txt_path": save_path,
            }
        # print(meta_info_subtask)
        organize_in_style2(meta_info=meta_info_subtask)


def get_ASOCA(save_path):
    meta_info = {
        "dataset_name": "ASOCA",
        "dataset_id": "51",
        "modality": "CT",
        "home_path": HOME_PATH,
        "dirpath": "51_ASOCA",
        "save_txt_path": save_path,
    }
    names = glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "image/*.nii.gz"))
    names.sort()
    label_names = [p.replace("/image", "/label") for p in names]
    # print(os.path.join(meta_info['home_path'], meta_info['dirpath'], "image/*.nii.gz")
    # print("debug ,", names)
    organize_by_names(names, label_names, meta_info=meta_info)
    
def get_BCV_Cervix(save_path):
    meta_info = {
        "dataset_name": "BCV-Cervix",
        "dataset_id": "52",
        "modality": "CT",
        "home_path": HOME_PATH,
        "dirpath": "52_BCV-Cervix/Training/",
        "save_txt_path": save_path,
    }
    names = glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'],  "img/*.nii.gz"))
    names.sort()
    label_names = [p.replace("/img/", "/label/").replace("-Image", "-Mask") for p in names]
    organize_by_names(names, label_names, meta_info=meta_info)

def get_NIHPancrease(save_path):
    meta_info = {
        "dataset_name": "NIHPancrease",
        "dataset_id": "53",
        "modality": "CT",
        "home_path": HOME_PATH,
        "dirpath": "53_NIHPancrease",
        "save_txt_path": save_path,
    }
    names = glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "data/*.nii.gz") )
    names.sort()
    label_names = [p.replace("/data/PANCREAS_", "/label/label") for p in names]
    organize_by_names(names, label_names, meta_info=meta_info)

def get_CTPelvic(save_path):
    meta_info = {
        "dataset_name": "CTPelvic1K",
        "dataset_id": "54",
        "modality": "CT",
        "home_path": HOME_PATH,
        "dirpath": "54_CTPelvic1K",
        "save_txt_path": save_path,
    }
    names = []
    names += glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "CTPelvic1K_dataset1_data/*.nii.gz"))
    names += glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "CTPelvic1K_dataset2_data/*.nii.gz"))
    names += glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "CTPelvic1K_dataset3_data/*.nii.gz"))
    names += glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "CTPelvic1K_dataset4_data/*.nii.gz"))
    names += glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "CTPelvic1K_dataset5_data/*.nii.gz"))
    names += glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "CTPelvic1K_dataset7_data/*.nii.gz"))
    names.sort()
    # xx_data.nii.gz   xx_mask_4label.nii.gz
    label_names = [p.replace("_data/", "_mask/").replace("_data.nii.gz", "_mask_4label.nii.gz") for p in names]
    organize_by_names(names, label_names, meta_info=meta_info)

def get_FLARE(save_path):
    meta_info = {
        "dataset_name": "FLARE",
        "dataset_id": "55",
        "modality": "CT",
        "home_path": HOME_PATH,
        "dirpath": "55_FLARE22Train",
        "save_txt_path": save_path,
        "class": ['liver', 'right kidney', 'spleen', 'pancrease', 'aorta','postcava','right adrenal gland','left darenal gland','gallbladder','esophagus','stomach','duodenum','left kidney'],
    }
    organize_in_nnunet_style(meta_info=meta_info)

# def get_HAN(save_path):
#     meta_info = {
#         "dataset_name": "Head-and-neck",
#         "dataset_id": "56",
#         "modality": "CT",
#         "home_path": HOME_PATH,
#         "dirpath": "56_Head-and-Neck-challenge",
#         "save_txt_path": save_path,
#     }
#     names = glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "data/*.nii.gz"))
#     names.sort()
#     label_names = [p.replace("/data/", "/label/") for p in names]
#     organize_by_names(names, label_names, meta_info=meta_info)

# def get_StructSeg(save_path):
#     meta_info = {
#         "dataset_name": "StructSeg2019",
#         "dataset_id": "57",
#         "modality": "CT",
#         "home_path": HOME_PATH,
#         "dirpath": "57_StructSeg",
#         "save_txt_path": save_path,
#     }
#     names = glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "HaN_OAR/data/*"))
#     names = [f"{name}/data.nii.gz" for name in names]
#     names.sort()
#     label_names = [p.replace("/data.nii.gz", "/label.nii.gz") for p in names]
#     organize_by_names(names, label_names, meta_info=meta_info)

def get_CHAOS(save_path):
    meta_info = {
        "dataset_name": "CHAOS",
        "dataset_id": "58",
        "modality": "MRI",
        "home_path": HOME_PATH,
        "dirpath": "58_CHAOST2/chaos_MR_T2_normalized/",
        "save_txt_path": save_path,
        "class": ["liver", "right kidney", "left kidney", "spleen"],
    }
    names = glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "image*.nii.gz"))
    names.sort()
    label_names = [p.replace("/image_", "/label_") for p in names]
    organize_by_names(names, label_names, meta_info=meta_info)

def get_SABS(save_path):
    meta_info = {
        "dataset_name": "SABS", # BTCV ?
        "dataset_id": "59",
        "modality": "CT",
        "home_path": HOME_PATH,
        "dirpath": "59_SABS/sabs_CT_normalized/",
        "save_txt_path": save_path,
        "class": ["spleen", "right kidney", "left kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "postcava", "portal vein and splenic vein", "pancrease", "right adrenal gland", "left adrenal gland"],
    }
    names = glob.glob(os.path.join(meta_info['home_path'], meta_info['dirpath'], "image_*.nii.gz"))
    names.sort()
    label_names = [p.replace("/image_", "/label_") for p in names]
    organize_by_names(names, label_names, meta_info=meta_info)    
    

def get_Totalseg(save_path):
    meta_info = {
        "dataset_name": "Totalseg",
        "dataset_id": "60",
        "modality": "CT",
        "home_path": HOME_PATH,
        # "dirpath": "nnUNet_raw/Dataset101_Totalseg",
        "dirpath": "60_Totalseg",
        "save_txt_path": save_path,
        "class": [],
    }
    organize_in_nnunet_style(meta_info=meta_info)


def get_WORD(save_path):
    meta_info = {
        "dataset_name": "WORDs", # BTCV ?
        "dataset_id": "07",
        "modality": "CT",
        "home_path": HOME_PATH,
        "dirpath": "07_WORD/WORD-V0.1.0/",
        "save_txt_path": save_path,
    }
    organize_in_style2(meta_info=meta_info)

def generate_all():
    save_path="./datasets/dataset_list/all_train.txt"
    clear_files(save_path)
    get_BCV_Abdomen(save_path)
    get_AbdomenCT_1K(save_path)
    get_AMOS(save_path)
    get_MSD(save_path)
    # get_ASOCA()
    # get_BCV_Cervix()
    # # get_NIHPancrease() # bug in data ?
    # get_CTPelvic()
    # get_FLARE()
    # get_SABS()

def generate_their():
    save_path="./datasets/dataset_list/their_train.txt"
    clear_files(save_path)
    save_path="./datasets/dataset_list/their_train.txt"
    get_BCV_Abdomen(save_path)
    get_AbdomenCT_1K(save_path)
    get_AMOS(save_path)
    get_MSD(save_path)

def generate_ours():
    save_path="./datasets/dataset_list/ours_train.txt"
    get_ASOCA(save_path)
    get_BCV_Cervix(save_path)
    # get_NIHPancrease() # bug in data ?
    get_CTPelvic(save_path)
    get_FLARE(save_path)
    get_SABS(save_path)

def generate_alp_dataset():
    save_path = "./datasets/dataset_list/alp_train.txt"
    clear_files(save_path)
    get_SABS(save_path)
    get_CHAOS(save_path)


if __name__ == "__main__":
    print(__file__)
    # generate_alp_dataset()
    save_path ="./datasets/dataset_list/totalseg_train.txt"
    clear_files(save_path)
    get_Totalseg(save_path)

    # save_path="./datasets/dataset_list/word_train.txt"
    print("Over")