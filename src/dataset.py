import math
import os.path
import numpy as np
import platform
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import torch.nn as nn
import random
import cv2
import pickle as pkl
import sys
from PIL import Image
import pandas as pd
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
sys.path.append(".")
# random.seed(50)
action_label_map={
            "jump rope":0, "jump":1, "jumping Jack":2, "sit on a chair and stand up":3, "lunge":4,
            "bow":5, "standing forward bend":6, "drink water":7, "kick":8, "throw":9,
            "wave hand":10, "golf swing":11, "march in place":12, "jog in place":13,  "swing arm":14,
            "stand still":15, "checking time":16, "clapping hand":17, "crouch":18, "squat":19,
            "pick up an object":20,  "hip rotation":21, "hula-hooping":22, "rotate head":23, "making a phone call":24,
            "walking counter-clockwise":25, "walking clockwise":26, "jog counter-clockwise":27, "jog clockwise":28, "walk forward and turn around":29,
            "jog forward and turn around":30, "side walk":31, "jump to the side":32,
            "jump forward and back": 33, "torso rotation": 34, "raise one arm to the front": 35, "raise one arm to the side": 36,
            "high leg lift": 37, "raise one leg to the side": 38, "turn 90 degree right or left then back": 39,
            "walking forward and backward": 40, "punch": 41, "empty42": 42, "Drawing a circle": 43, "Drawing an X": 44, "Mopping the floor": 45,
            "push hand to the front": 46, "sitting still": 47, "swiping left and right": 48, "stretching leg to the side": 49,
            "wake up stretch": 50, "playing phone": 51, "nodding head": 52, "empty53":53, "playing table tennis": 54, "baseball pitch": 55,
            "shooting basketball": 56, "bowling": 57, "walking to the side and turn back": 58, "side kick": 59, "kick soccer ball": 60,
            "bend torso to side": 61, "jumping lunge": 62, "swing arm like throwing": 63, "step over obstacle and back": 64,
            "rotating head": 65
        }

elimination_list=[
    "windmills both arms from shoulder",
    "moving upper body in circles and step forward and moving left ankle in circles",
    "stretching the right leg and rotating the right leg",
    "arms move forward and backward",
    "standing and moves hands around at chest",
    "place",
    "wiggle waist and kick out left leg and duck down",
    "standing and playing the piano",
    "bounces both hands up and down quickly near chest while rotating torso slightly and slaps near left side of head with left hand",
    "march forward and standing and march back",
    "hand gesture series a and hand gesture series b and hand gesture c",
    "walk and walk backwards",
    "overhead arms stretch and swing arms and calf raise",
    "t-pose and lowering arms and hopping on right foot",
    "steps backwards and walks forwards",
    "spin spin arms backwards and bend forward",
    "move around moving legs kicking and turn around",
    "moving arms",
    "side-stepping on left and side-stepping on right",
    "walking toward right. walking backward",
]

def load_sync_list(infilename):
    sync_list = []
    csv_file = pd.read_csv(infilename, header=0)
    print(infilename)
    for row_index in range(csv_file.shape[0]):
        trail_name = csv_file["trial_name"][row_index]
        sync_list.append(trail_name)
    return sync_list

def convert_image_to_rgb(image):
    return image.convert("RGB")

trial_folder_map={}
for trial in load_sync_list("./data/local_data/data_0505/sync_file.csv"):
    trial_folder_map[trial]="data_0505"
#for trial in load_sync_list("./data/local_data/data_0512/sync_file.csv"):
#    trial_folder_map[trial]="data_0512"
#for trial in load_sync_list("./data/local_data/data_0512_0318/sync_file.csv"):
#    trial_folder_map[trial]="data_0512_0318"


def read_gpt_data(file_path):
    description_list=[]
    num_line=0
    with open(file_path, 'r') as infile:
        for line in infile.readlines():
            try:
                act_name=line.split('.')[0]
                des_1 = line.split('.')[1]
                des_2 = line.split('.')[2]
                des_3 = line.split('.')[3]
                des_4 = line.split('.')[4]
                des_5 = line.split('.')[5]
                description_list.append([act_name, des_1, des_2, des_3, des_4, des_5])
            except:
                pass
                # print(num_line, line.replace("\n", ""))
            num_line+=1
    return description_list
def remove_duplicates(list):
    unique_list = []
    start_list=[]
    end_list=[]
    unique_list.append(list[0].lower())
    start_list.append(0)
    for i in range(1, len(list)):
        if list[i].lower()!=list[i-1].lower():
            unique_list.append(list[i].lower())
            end_list.append(i)
            start_list.append(i)
    end_list.append(len(list))
    return unique_list, np.array(end_list)-np.array(start_list)

class babel_dataset(Dataset):
    def __init__(self, data_paths=["./exp7-8/4_td_hms/"], labe_dict_path="./exp7-8/1_segs_label_dict/",
                 dataset_list=["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT"],
                 crop_size=(224,256), img_size=(224,224), aug_ratio=1, if_range_aug=True):
        self.crop_size = crop_size
        self.img_size = img_size
        self.hm_text_list=[]
        self.label_dict={}
        self.hm_td_paths=[]
        self.stft_win_length = 256
        self.stft_hop_length = 16
        self.aug_ratio=aug_ratio
        self.if_range_aug=if_range_aug
        for dataset_name in dataset_list:
            with open("./{}/{}.pkl".format(labe_dict_path, dataset_name), "rb") as f1:
                self.label_dict[dataset_name]=pkl.load(f1)
            for data_path in data_paths:
                self.hm_td_paths.extend(glob.glob("{}/{}/*_td.npy".format(data_path, dataset_name)))
        print("online dataset_length:", len(self.hm_td_paths))

    def __getitem__(self, index):
        hm_td_path=self.hm_td_paths[int(index//self.aug_ratio)]
        dataset_name=hm_td_path.split("/")[-2]
        file_name=hm_td_path.split("/")[-1][:-18]

        hm_td = np.load(hm_td_path)
        hm_tr = np.load(hm_td_path.replace("_td.npy", "_tr.npy"))
        hm_ta = np.load(hm_td_path.replace("_td.npy", "_ta.npy"))
        crop_size=self.crop_size[1]
        if hm_td.shape[1] < self.crop_size[1]:
            crop_size = hm_td.shape[1]

        # print(file_name, hm_td.shape[1])
        rand_int=random.randint(0,hm_td.shape[1]-crop_size)
        ##calculate labels
        start_time=rand_int*self.stft_hop_length/128
        end_time=(rand_int+ crop_size)*(self.stft_hop_length/128)
        frame_raw_text=self.label_dict[dataset_name]["frame_raw_text_dict"][file_name][math.ceil(start_time):int(end_time)]
        # unique_frame_raw_text=set(frame_raw_text)
        unique_frame_raw_text = list(dict.fromkeys(list(frame_raw_text)))
        if "transition" in unique_frame_raw_text:
            unique_frame_raw_text.remove("transition")
        unique_frame_raw_text=" and ".join(unique_frame_raw_text)
        # print(len(ori_frame_raw_text), math.ceil(start_time), int(end_time), unique_frame_raw_text)
        crop_img_td = hm_td[128 - self.crop_size[0] // 2:128 + self.crop_size[0] // 2, rand_int:  rand_int+ self.crop_size[1]]
        crop_img_td = cv2.resize(crop_img_td, self.img_size)
        crop_img_td = 2 * (crop_img_td - np.min(crop_img_td)) / (np.max(crop_img_td) - np.min(crop_img_td)) - 1
        crop_img_tr = hm_tr[:, rand_int:rand_int+ self.crop_size[1]]
        if self.if_range_aug:
            rand_int_range = random.randint(8, 15)
            crop_img_tr =np.concatenate([np.repeat(crop_img_tr[0:1], rand_int_range, 0), crop_img_tr[:-rand_int]], axis=0)
        crop_img_tr = cv2.resize(crop_img_tr, self.img_size)
        crop_img_tr = 2*(crop_img_tr - np.min(crop_img_tr)) / (np.max(crop_img_tr) - np.min(crop_img_tr))-1
        crop_img_ta = hm_ta[:, rand_int:rand_int+ self.crop_size[1]]
        crop_img_ta = cv2.resize(crop_img_ta, self.img_size)
        crop_img_ta = 2*(crop_img_ta - np.min(crop_img_ta)) / (np.max(crop_img_ta) - np.min(crop_img_ta))-1
        crop_img=np.stack((crop_img_td, crop_img_tr, crop_img_ta), axis=0)
        # return crop_img, unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text
        return (crop_img,
                unique_frame_raw_text,
                [unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text],
                unique_frame_raw_text)
    def __len__(self):
        return len(self.hm_td_paths)*self.aug_ratio


humanml3d_elimination_hm=[
    "in there is a stand behind it",
    "the man begins standing still, he lifts and kicks his right leg and opposite arm than switches back and forth three times, he than goes back to his beginning position",
    "person stands and raises right leg, then proceeds to kick into the air towards the right, person then pivots their torso tot he right, raising their left leg and kicking to the right, then comes back to stand center",
]
class HumanML3DDataset(Dataset):
    def __init__(self, data_paths=["./exp7-8-3/4_td_hms/"],
                 text_paths="./HumanML3D/HumanML3D/texts",
                 gpt_data_location="./exp7-8-3/6_prompts",
                 csv_path="./data/humanml3d_data/index.csv",
                 dataset_list=["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT"],
                 crop_size=(224,256), img_size=(224,224), aug_ratio=1, if_use_gpt=True, if_range_aug=True):
        self.crop_size = crop_size
        self.img_size = img_size
        self.hm_text_list=[]
        self.hm_td_paths=[]
        self.stft_win_length = 256
        self.stft_hop_length = 16
        self.aug_ratio=aug_ratio
        self.sync_map=self.load_sync_map(csv_path)
        self.text_paths=text_paths
        self.if_range_aug = if_range_aug
        self.if_use_gpt=if_use_gpt



        for dataset_name in dataset_list:
            for data_path in data_paths:
                # print("{}/{}/*_td.npy".format(data_path, dataset_name))
                self.hm_td_paths.extend(glob.glob("{}/{}/*_td.npy".format(data_path, dataset_name)))

        gpt_text_file_paths = glob.glob("{}/humanml3d_sample_00_00.txt".format(gpt_data_location))
        self.description_list = []  # [[act_name, des_1, des_2, des_3, des_4], [], ...]
        for gpt_text_file_path in gpt_text_file_paths:
            single_description_list = read_gpt_data(gpt_text_file_path)
            self.description_list.extend(single_description_list)
        self.description_map = {}
        for description in self.description_list:
            if description[0] not in self.description_map.keys():
                self.description_map[description[0]] = []
                self.description_map[description[0]].append(description[1:])
            else:
                self.description_map[description[0]].append(description[1:])


        for hm_td_path in self.hm_td_paths:
            dataset_name = hm_td_path.split("/")[-1].split("__")[0]
            subject = hm_td_path.split("/")[-1].split("__")[1]
            action = hm_td_path.split("/")[-1].split("__")[2]
            text_file = self.sync_map["{}/{}/{}/{}.npy".format("./pose_data", dataset_name, subject, action)]
            unique_frame_raw_text = self.read_text_label(
                "{}/{}".format(self.text_paths, text_file.replace("npy", "txt")))
            unique_frame_raw_text = unique_frame_raw_text.replace(".", ",")
            while unique_frame_raw_text[-1] == ",":
                unique_frame_raw_text = unique_frame_raw_text[:-1]
            if unique_frame_raw_text in humanml3d_elimination_hm:
                self.hm_td_paths.remove(hm_td_path)
            #     continue
            # if unique_frame_raw_text not in self.description_map.keys():
            #     print(unique_frame_raw_text)

        # quit()


        print("online dataset_length:", len(self.hm_td_paths)*self.aug_ratio)

    def read_text_label(self, text_file):
        text_label = []
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                text = line.split("#")[0]
                text_label.append(text)
        ## random return a text label
        # randint=np.random.randint(0, len(text_label))
        randint=0
        return text_label[randint]
    def load_sync_map(self, infilename):
        sync_map = {}
        csv_file = pd.read_csv(infilename, header=0)
        print(infilename)
        for row_index in range(csv_file.shape[0]):
            file_name = csv_file["source_path"][row_index]
            new_name = csv_file["new_name"][row_index]
            sync_map[file_name]=new_name
        return sync_map
    def __getitem__(self, index):
        hm_td_path=self.hm_td_paths[int(index//self.aug_ratio)]
        dataset_name = hm_td_path.split("/")[-1].split("__")[0]
        subject = hm_td_path.split("/")[-1].split("__")[1]
        action = hm_td_path.split("/")[-1].split("__")[2]

        hm_td = np.load(hm_td_path)
        hm_tr = np.load(hm_td_path.replace("_td.npy", "_tr.npy"))
        hm_ta = np.load(hm_td_path.replace("_td.npy", "_ta.npy"))
        crop_size=self.crop_size[1]
        if hm_td.shape[1] < self.crop_size[1]:
            crop_size = hm_td.shape[1]

        rand_int=random.randint(0,hm_td.shape[1]-crop_size)
        text_file = self.sync_map["{}/{}/{}/{}.npy".format("./pose_data", dataset_name, subject, action)]
        unique_frame_raw_text = self.read_text_label(
            "{}/{}".format(self.text_paths, text_file.replace("npy", "txt")))

        unique_frame_raw_text = unique_frame_raw_text.replace(".", ",")
        while unique_frame_raw_text[-1] == ",":
            unique_frame_raw_text =unique_frame_raw_text[:-1]


        if self.if_use_gpt:
            detailed_des=self.description_map[unique_frame_raw_text]
            text_candidate_num=len(detailed_des)
            text_candiate_index=random.randint(0,text_candidate_num-1)

        crop_img_td = hm_td[128 - self.crop_size[0] // 2:128 + self.crop_size[0] // 2, rand_int:  rand_int+ self.crop_size[1]]
        crop_img_td = cv2.resize(crop_img_td, self.img_size)
        crop_img_td = 2 * (crop_img_td - np.min(crop_img_td)) / (np.max(crop_img_td) - np.min(crop_img_td)) - 1
        crop_img_tr = hm_tr[:, rand_int:rand_int+ self.crop_size[1]]
        if self.if_range_aug:
            rand_int_range = random.randint(8, 15)
            crop_img_tr =np.concatenate([np.repeat(crop_img_tr[0:1], rand_int_range, 0), crop_img_tr[:-rand_int]], axis=0)
        crop_img_tr = cv2.resize(crop_img_tr, self.img_size)
        crop_img_tr = 2*(crop_img_tr - np.min(crop_img_tr)) / (np.max(crop_img_tr) - np.min(crop_img_tr))-1
        crop_img_ta = hm_ta[:, rand_int:rand_int+ self.crop_size[1]]
        crop_img_ta = cv2.resize(crop_img_ta, self.img_size)
        crop_img_ta = 2*(crop_img_ta - np.min(crop_img_ta)) / (np.max(crop_img_ta) - np.min(crop_img_ta))-1
        crop_img=np.stack((crop_img_td, crop_img_tr, crop_img_ta), axis=0)

        if self.if_use_gpt:
            return (crop_img,
                    detailed_des[text_candiate_index],
                    detailed_des[text_candiate_index],
                    detailed_des[text_candiate_index])


        return (crop_img,
                unique_frame_raw_text,
                [unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text],
                unique_frame_raw_text)
    def __len__(self):
        return len(self.hm_td_paths)*self.aug_ratio

class babel_dataset_gpt(Dataset):
    def __init__(self, data_paths=["./exp7-8/4_td_hms/"], label_dict_path="./exp7-8/1_segs_label_dict/",
                 dataset_list=["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT"],
                 gpt_data_location="./exp7-8/6_prompts",
                 crop_size=(224,256), img_size=(224,224), if_range_aug=True, if_use_gpt=True,
                 if_use_img=False, aug_ratio=1):
        self.crop_size = crop_size
        self.img_size = img_size
        self.hm_text_list=[]
        self.label_dict={}
        self.hm_td_paths=[]
        self.stft_win_length = 256
        self.stft_hop_length = 16
        self.if_range_aug=if_range_aug
        self.if_use_gpt=if_use_gpt
        self.if_use_img=if_use_img
        self.aug_ratio = aug_ratio
        for dataset_name in dataset_list:
            with open("./{}/{}.pkl".format(label_dict_path, dataset_name), "rb") as f1:
                self.label_dict[dataset_name]=pkl.load(f1)
                frame_raw_text_dict = self.label_dict[dataset_name]["frame_raw_text_dict"]
                for file_name, frame_raw_text in frame_raw_text_dict.items():
                    if_skip_flag = False
                    unique_frame_raw_text, time = remove_duplicates(list(frame_raw_text))
                    for i, item in enumerate(unique_frame_raw_text):
                        if item=="transition" and time[i]>=20:
                            if_skip_flag=True
                            break
                    if if_skip_flag:
                        continue
                    while "transition" in unique_frame_raw_text:
                        unique_frame_raw_text.remove("transition")
                    if len(unique_frame_raw_text) == 0:
                        continue
                    unique_frame_raw_text, _ = remove_duplicates(unique_frame_raw_text)
                    if len(unique_frame_raw_text) > 3:  ##can try different numbers
                        continue
                    ## manual filter
                    cur_combinations=[]
                    for i in range(1, len(unique_frame_raw_text) + 1):
                        for j in range(len(unique_frame_raw_text) - i + 1):
                            cur_combinations.append(" and ".join(unique_frame_raw_text[j:j+i]))
                    if len(set(cur_combinations).intersection(set(elimination_list)))!=0:
                        # print(set(cur_combinations).intersection(set(elimination_list)))
                        continue

                    for data_path in data_paths:
                        if os.path.exists("{}/{}/{}__000degree_td.npy".format(data_path,dataset_name, file_name)):
                            self.hm_td_paths.append("{}/{}/{}__000degree_td.npy".format(data_path,dataset_name, file_name))
                        else:
                            print("not used","{}/{}/{}__000degree_td.npy".format(data_path,dataset_name, file_name))
        print("online dataset_length:", len(self.hm_td_paths)*self.aug_ratio)
        gpt_text_file_paths=glob.glob("{}/babel_sample_04_03.txt".format(gpt_data_location))
        self.description_list=[]#[[act_name, des_1, des_2, des_3, des_4], [], ...]
        for gpt_text_file_path in gpt_text_file_paths:
            single_description_list=read_gpt_data(gpt_text_file_path)
            self.description_list.extend(single_description_list)
        self.description_map={}
        for description in self.description_list:
            # description[0] = the activity label, description[1:] = gpt description for activity
            if description[0] not in self.description_map.keys():
                self.description_map[description[0]]=[]
                self.description_map[description[0]].append(description[1:])
            else:
                self.description_map[description[0]].append(description[1:])
    def __getitem__(self, index):
        hm_td_path=self.hm_td_paths[int(index//self.aug_ratio)]
        dataset_name=hm_td_path.split("/")[-2]
        file_name=hm_td_path.split("/")[-1][:-18]

        hm_td = np.load(hm_td_path)
        hm_tr = np.load(hm_td_path.replace("_td.npy", "_tr.npy"))
        hm_ta = np.load(hm_td_path.replace("_td.npy", "_ta.npy"))
        crop_size=self.crop_size[1]
        if hm_td.shape[1] < self.crop_size[1]:
            crop_size = hm_td.shape[1]

        # print(file_name, hm_td.shape[1])
        rand_int = random.randint(0, hm_td.shape[1]-crop_size)
        ##calculate labels
        start_time=rand_int*self.stft_hop_length/128
        end_time=(rand_int+ crop_size)*(self.stft_hop_length/128)

        frame_raw_text=self.label_dict[dataset_name]["frame_raw_text_dict"][file_name][math.ceil(start_time):int(end_time)]
        # unique_frame_raw_text=set(frame_raw_text)
        unique_frame_raw_text,_ = remove_duplicates(list(frame_raw_text))
        while "transition" in unique_frame_raw_text:
            unique_frame_raw_text.remove("transition")
        if len(unique_frame_raw_text)==0:
            print(file_name)
        
        unique_frame_raw_text, _ = remove_duplicates(unique_frame_raw_text)
        unique_frame_raw_text=" and ".join(unique_frame_raw_text).lower()
        ## now find the mapping
        if self.if_use_gpt:
            # these are the descriptions for each label within the frame, choose one at random as the representative for the frame
            detailed_des=self.description_map[unique_frame_raw_text]
            text_candidate_num=len(detailed_des)
            text_candiate_index=random.randint(0,text_candidate_num-1)
        
        # as long as the crop size is (256, *), the first piece of indexing will evaluate to 0:256 (the full height of the heatmap)
        crop_img_td = hm_td[128 - self.crop_size[0] // 2:128 + self.crop_size[0] // 2, rand_int:  rand_int+ self.crop_size[1]]
        crop_img_td = cv2.resize(crop_img_td, self.img_size)
        crop_img_td = 2 * (crop_img_td - np.min(crop_img_td)) / (np.max(crop_img_td) - np.min(crop_img_td)) - 1  # normalize heatmap to values [-1, 1]

        crop_img_tr = hm_tr[:, rand_int:rand_int+ self.crop_size[1]]
        if self.if_range_aug:
            rand_int_range = random.randint(8, 15)
            crop_img_tr =np.concatenate([np.repeat(crop_img_tr[0:1], rand_int_range, 0), crop_img_tr[:-rand_int]], axis=0)
        crop_img_tr = cv2.resize(crop_img_tr, self.img_size)
        crop_img_tr = 2*(crop_img_tr - np.min(crop_img_tr)) / (np.max(crop_img_tr) - np.min(crop_img_tr))-1

        crop_img_ta = hm_ta[:, rand_int:rand_int+ self.crop_size[1]]
        crop_img_ta = cv2.resize(crop_img_ta, self.img_size)
        crop_img_ta = 2*(crop_img_ta - np.min(crop_img_ta)) / (np.max(crop_img_ta) - np.min(crop_img_ta))-1

        crop_img=np.stack((crop_img_td, crop_img_tr, crop_img_ta), axis=0)

        if self.if_use_img:
            ##get the most closeted rendered image
            pos = round((end_time + start_time) / 20 - 0.5)
            img_list = []
            for i in range(-1, 2):
                rendered_img_path = (
                    "./exp7-8-2/0_rendered_img/{}/{}__{:02d}.jpeg".format(dataset_name, file_name, pos + i))
                rendered_img = Image.open(rendered_img_path)
                img_list.append(rendered_img)

        if self.if_use_gpt and self.if_use_img:
            return (crop_img,
                    img_list,
                    detailed_des[text_candiate_index],
                    detailed_des[text_candiate_index])
        if self.if_use_gpt:
            return (crop_img,  # heatmaps
                    detailed_des[text_candiate_index],  # activity label descriptions
                    detailed_des[text_candiate_index],
                    detailed_des[text_candiate_index])
        if self.if_use_gpt==False and self.if_use_img:
            return (crop_img,
                    img_list,
                    [unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text,unique_frame_raw_text],
                    unique_frame_raw_text)

        return (crop_img,
                unique_frame_raw_text,
                [unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text, unique_frame_raw_text],
                unique_frame_raw_text)
    def __len__(self):
        return len(self.hm_td_paths)*self.aug_ratio


class local_dataset(Dataset):
    def __init__(self, trial_list, query_classes,
                 data_location="./Radar_data_collection/Mar_2024/data_0413/hm_dra_modified_length_0-100/",
                 gpt_data_location="./Radar_data_collection/Mar_2024/gpt_data_0505/",
                 crop_size=(224,384), img_size=(224,224),
                 ratio=1.0, order="left_part", sampling_gap=8, if_range_aug=False, num_hm_segs_per_activity=1, use_adjacent_hm=False):
        self.if_range_aug=if_range_aug

        self.ori_label_2gt = {}
        for i, ori_label in enumerate(query_classes):
            self.ori_label_2gt[ori_label]=i

        self.trial_list = trial_list
        self.trial_size = 0
        self.raw_location = data_location
        if self.raw_location.split("/")[1] == "Video_data_collection":
            self.if_range_aug=True
        self.img_label_list=[]
        self.sampling_gap=sampling_gap
        self.img_size =img_size
        self.crop_size = crop_size
        self.action_label_map = action_label_map
        self.action_label_map_inv = {v: k for k, v in self.action_label_map.items()}
        gpt_text_file_paths=glob.glob("{}/local_sample_02_no001.txt".format(gpt_data_location))
        self.num_hm_segs_per_activity = num_hm_segs_per_activity
        self.use_adjacent_hm = use_adjacent_hm
        self.description_list=[]#[[act_name, des_1, des_2, des_3, des_4], [], ...]
        for gpt_text_file_path in gpt_text_file_paths:
            single_description_list=read_gpt_data(gpt_text_file_path)
            self.description_list.extend(single_description_list)

        self.description_map={}
        for description in self.description_list:
            if description[0] not in self.description_map.keys():
                self.description_map[description[0]]=[]
                self.description_map[description[0]].append(description[1:])
            else:
                self.description_map[description[0]].append(description[1:])
        self.inference_description_list=[]
        for description in self.description_map.values():#[[[des_1, des_2, des_3, des_4],[]], [[],[],[]]]
            self.inference_description_list.append(description[0])
        for trial_name in trial_list:#["Sxx", ""]
            folder=trial_folder_map[trial_name]
            if int(trial_name[6: 9]) not in query_classes:
                continue
            label = self.ori_label_2gt[int(trial_name[6:9])]
            text=self.description_map[self.action_label_map_inv[int(trial_name[6:9])]]##use detailed des
            # text=[[self.action_label_map_inv[int(trial_name[6:9])]]*5]##use short des
            # text = [[self.description_map[self.action_label_map_inv[int(trial_name[6:9])]][0][0]]*5]##use short des
            if self.raw_location.split("/")[1] == "data":
                path_td = self.raw_location.replace("data_0505",folder) + "/{}/time_dop.npy".format(trial_name)
                path_tr = self.raw_location.replace("data_0505",folder) + "/{}/time_range.npy".format(trial_name)
                path_ta = self.raw_location.replace("data_0505",folder) + "/{}/time_angle.npy".format(trial_name)
                img_ori_td = np.load(path_td)[:, 200:]
                img_ori_tr = np.load(path_tr)[:, 200:]
                img_ori_ta = np.load(path_ta)[:, 200:]
            elif self.raw_location.split("/")[1] == "Video_data_collection":
                try:
                    folder = "{}_{}".format(folder.split("_")[1], folder.split("_")[0])
                    path_td = glob.glob("{}/{}*_td.npy".format(self.raw_location.replace("0505_data",folder), trial_name))[0]
                    path_tr = glob.glob("{}/{}*_tr.npy".format(self.raw_location.replace("0505_data",folder), trial_name))[0]
                    path_ta = glob.glob("{}/{}*_ta.npy".format(self.raw_location.replace("0505_data",folder), trial_name))[0]
                    img_ori_td = np.load(path_td)[:, 200:]
                    img_ori_tr = np.load(path_tr)[:, 200:]
                    img_ori_ta = np.load(path_ta)[:, 200:]
                except:
                    continue
            else:
                assert "Please use local from either Radar_data_collection or Video_data_collection"

            self.trial_size+=1
            img_len_ori = img_ori_td.shape[1]
            if order=="left_part":
                img_td = img_ori_td[:, 0:int(img_len_ori*ratio)]
                img_tr = img_ori_tr[:, 0:int(img_len_ori * ratio)]
                img_ta = img_ori_ta[:, 0:int(img_len_ori * ratio)]
            else:
                img_td = img_ori_td[:, int(img_len_ori*ratio):]
                img_tr = img_ori_tr[:, int(img_len_ori * ratio):]
                img_ta = img_ori_ta[:, int(img_len_ori * ratio):]
            self.img_label_list.append([img_td, img_tr, img_ta, text, label])
        self.img_time_length = self.img_label_list[0][0].shape[1]
        self.seg_number_per_trial = (self.img_time_length-self.img_size[1]+1) // self.sampling_gap  # number of segments possible per activity heatmap
        print("real dataset_length:",self.trial_size*self.seg_number_per_trial)

    def __getitem__(self, index):
        # During training, the dataloader randomizes the indices and gets the data using them
        # Try randomly choosing the second heatmap in the pair for improved diversity during training
        # Draw accuracy curve over iterations

        trial_no=index//self.seg_number_per_trial  # determines which heatmap to use (one heatmap can yield multiple segments)
        img_td, img_tr, img_ta, text, label = self.img_label_list[trial_no]
        assert len(text) == 1  # should be one text per heatmap (one activity being performed in the heatmap)
        # print("Local dataset hm shape: ", img_td.shape)
        text_candidate_num=len(text)
        text_candiate_index=random.randint(0,text_candidate_num-1)
        trial_index = index-trial_no*self.seg_number_per_trial  # where to start within the current activity given the index
        
        full_crop_img = np.array([])
        windows_used = []  # used for debugging
        num_hms_added = 0
        for seg_num in range(1, self.num_hm_segs_per_activity+1):
            if self.use_adjacent_hm:
                # + (seg_num * self.crop_size[1]) chooses the next adjacent heatmap segment
                hm_window = (trial_index*self.sampling_gap + ((seg_num-1)*self.crop_size[1]), 
                            trial_index*self.sampling_gap + ((seg_num-1)*self.crop_size[1] + self.crop_size[1]))
                if hm_window[1] > img_td.shape[1]:
                    shift = (hm_window[1] - img_td.shape[1]) // self.crop_size[1]  # handle if hm_window[1]*2 > img_td.shape[1] 
                    # use left adjacent hm segment instead
                    hm_window = (trial_index*self.sampling_gap + (seg_num*self.crop_size[1] - (self.crop_size[1] * (shift + seg_num+1))),
                                trial_index*self.sampling_gap + (seg_num*self.crop_size[1] - (self.crop_size[1] * (shift + seg_num))))
            else: # randomly selected hm window (theoretically, this should perform better, because there are more possible training pairs when using random)
                # (0, 2665) --> One activity
                if seg_num == 1:
                    # initial hm, should select window as if only selecting one hm segment
                    hm_window = (trial_index*self.sampling_gap, trial_index*self.sampling_gap + self.crop_size[1])
                else:
                    # random hm segment outside of initial hm window 
                    random_start_idx_left = random.randint(0, hm_window[0]-self.crop_size[1]-1) if hm_window[0]-self.crop_size[1]-1 > 0 else -1
                    random_start_idx_right = random.randint(hm_window[0]+self.crop_size[1], img_td.shape[1]-self.crop_size[1]) if hm_window[0]+self.crop_size[1] < img_td.shape[1]-self.crop_size[1] else -1
                    random_start_idx = random.choice([start for start in [random_start_idx_left, random_start_idx_right] if start != -1])
                    hm_window = (random_start_idx, random_start_idx+self.crop_size[1])
            num_hms_added += 1
            crop_img_td = img_td[128-self.crop_size[0]//2:128+self.crop_size[0]//2, hm_window[0]:hm_window[1]]
            crop_img_td = cv2.resize(crop_img_td, self.img_size)
            crop_img_td = 2*(crop_img_td - np.min(crop_img_td)) / (np.max(crop_img_td) - np.min(crop_img_td))-1

            ## need to do random tr augmentation here for sim, (also seen real?)
            ##sim top pad 10-20, bot crop 10-15

            crop_img_tr = img_tr[:, hm_window[0]:hm_window[1]]
            if self.if_range_aug:
                rand_int = random.randint(8, 15)
                crop_img_tr =np.concatenate([np.repeat(crop_img_tr[0:1], rand_int, 0), crop_img_tr[:-rand_int]], axis=0)
            crop_img_tr = cv2.resize(crop_img_tr, self.img_size)
            crop_img_tr = 2*(crop_img_tr - np.min(crop_img_tr)) / (np.max(crop_img_tr) - np.min(crop_img_tr))-1
            crop_img_ta = img_ta[:, hm_window[0]:hm_window[1]]
            crop_img_ta = cv2.resize(crop_img_ta, self.img_size)
            crop_img_ta = 2*(crop_img_ta - np.min(crop_img_ta)) / (np.max(crop_img_ta) - np.min(crop_img_ta))-1

            crop_img=np.stack((crop_img_td, crop_img_tr, crop_img_ta), axis=0)
            full_crop_img=np.append(full_crop_img, [crop_img], axis=0) if full_crop_img.shape[0] else np.array([crop_img])
            windows_used.append(hm_window)
        assert len(full_crop_img) == num_hms_added
        # print("Shape of cropped hms:", full_crop_img[0, :, :, :].shape)
        if full_crop_img.shape[0] == 1:
            return full_crop_img[0, :, :, :], text[text_candiate_index], text[text_candiate_index], [label]
        # text does not need to be altered when num_hm_segs_per_activity > 1 
        # first val is the originally used hm (hm used in vanilla mmclip), and the last val is the hms to optimize cosine sim over for intra-heatmap alignmetn
        return full_crop_img[0, :, :, :], text[text_candiate_index], text[text_candiate_index], [label], full_crop_img[1:, :, :, :]

    def __len__(self):
        return self.trial_size*self.seg_number_per_trial

class local_dataset_fs(Dataset):
    def __init__(self, trial_list, query_classes,
                 data_location="./Radar_data_collection/Mar_2024/data_0413/hm_dra_modified_length_0-100/",
                 gpt_data_location="./Radar_data_collection/Mar_2024/gpt_data_0413/",
                 crop_size=(224,384), img_size=(224,224),
                 ratio=1.0, order="left_part", sampling_gap=8, start_index=0, few_shot_sample=1, if_range_aug=False):
        self.ori_label_2gt = {}
        self.img_label_dict={}
        self.if_range_aug=if_range_aug
        for i, ori_label in enumerate(query_classes):
            self.ori_label_2gt[ori_label]=i
            self.img_label_dict[i]=[]
        self.trial_list = trial_list
        self.trial_size = 0
        self.raw_location = data_location
        if self.raw_location.split("/")[1] == "Video_data_collection":
            self.if_range_aug=True
        self.sampling_gap=sampling_gap
        self.img_size =img_size
        self.crop_size = crop_size
        self.few_shot_sample=few_shot_sample
        self.action_label_map = action_label_map
        self.action_label_map_inv = {v: k for k, v in self.action_label_map.items()}
        gpt_text_file_paths=glob.glob("{}/qiming_sample_02_no001.txt".format(gpt_data_location))
        self.description_list=[]#[[act_name, des_1, des_2, des_3, des_4], [], ...]
        for gpt_text_file_path in gpt_text_file_paths:
            single_description_list=read_gpt_data(gpt_text_file_path)
            self.description_list.extend(single_description_list)

        self.description_map={}
        for description in self.description_list:
            if description[0] not in self.description_map.keys():
                self.description_map[description[0]]=[]
                self.description_map[description[0]].append(description[1:])
            else:
                self.description_map[description[0]].append(description[1:])
        self.inference_description_list=[]
        for description in self.description_map.values():#[[des_1, des_2, des_3, des_4],[]]
            self.inference_description_list.append(description[0])
        for trial_name in trial_list:#["Sxx", ""]
            if int(trial_name[6: 9]) not in query_classes:
                continue
            label = self.ori_label_2gt[int(trial_name[6:9])]
            text=self.description_map[self.action_label_map_inv[int(trial_name[6:9])]]##use detailed des
            #text=[[self.action_label_map_inv[int(trial_name[4:6])]]*4]##use short des
            self.trial_size+=1
            if self.raw_location.split("/")[1] == "data":
                path_td = self.raw_location + "/{}/time_dop.npy".format(trial_name)
                path_tr = self.raw_location + "/{}/time_range.npy".format(trial_name)
                path_ta = self.raw_location + "/{}/time_angle.npy".format(trial_name)
                img_ori_td = np.load(path_td)[:, 200:]
                img_ori_tr = np.load(path_tr)[:, 200:]
                img_ori_ta = np.load(path_ta)[:, 200:]
            elif self.raw_location.split("/")[1] == "Video_data_collection":
                path_td = glob.glob("{}/{}*_td.npy".format(self.raw_location, trial_name))[0]
                path_tr = glob.glob("{}/{}*_tr.npy".format(self.raw_location, trial_name))[0]
                path_ta = glob.glob("{}/{}*_ta.npy".format(self.raw_location, trial_name))[0]
                img_ori_td = np.load(path_td)[:, 200:]
                img_ori_tr = np.load(path_tr)[:, 200:]
                img_ori_ta = np.load(path_ta)[:, 200:]
            else:
                assert "Please use local from either Radar_data_collection or Video_data_collection"

            img_len_ori = img_ori_td.shape[1]
            if order=="left_part":
                img_td = img_ori_td[:, 0:int(img_len_ori*ratio)]
                img_tr = img_ori_tr[:, 0:int(img_len_ori * ratio)]
                img_ta = img_ori_ta[:, 0:int(img_len_ori * ratio)]
            else:
                img_td = img_ori_td[:, int(img_len_ori*ratio):]
                img_tr = img_ori_tr[:, int(img_len_ori * ratio):]
                img_ta = img_ori_ta[:, int(img_len_ori * ratio):]
            for sample_id in range(self.few_shot_sample):
                crop_img_td = img_td[128 - self.crop_size[0] // 2:128 + self.crop_size[0] // 2, start_index+sample_id*sampling_gap:start_index+sample_id*sampling_gap+crop_size[1]]
                crop_img_td = cv2.resize(crop_img_td, self.img_size)
                crop_img_td = 2 * (crop_img_td - np.min(crop_img_td)) / (np.max(crop_img_td) - np.min(crop_img_td)) - 1
                crop_img_tr = img_tr[:, start_index+sample_id*sampling_gap:start_index+sample_id*sampling_gap+crop_size[1]]
                if self.if_range_aug:
                    rand_int = random.randint(8, 15)
                    crop_img_tr = np.concatenate([np.repeat(crop_img_tr[0:1], rand_int, 0), crop_img_tr[:-rand_int]],
                                                 axis=0)
                crop_img_tr = cv2.resize(crop_img_tr, self.img_size)
                crop_img_tr = 2 * (crop_img_tr - np.min(crop_img_tr)) / (np.max(crop_img_tr) - np.min(crop_img_tr)) - 1
                crop_img_ta = img_ta[:, start_index+sample_id*sampling_gap:start_index+sample_id*sampling_gap+crop_size[1]]
                crop_img_ta = cv2.resize(crop_img_ta, self.img_size)
                crop_img_ta = 2 * (crop_img_ta - np.min(crop_img_ta)) / (np.max(crop_img_ta) - np.min(crop_img_ta)) - 1
                crop_img = np.stack((crop_img_td, crop_img_tr, crop_img_ta), axis=0)
                self.img_label_dict[label].append(crop_img)
        self.img_label_list=[]
        for label, hm_list in  self.img_label_dict.items():
            hm_arr=np.stack(hm_list)#b*c*h*w
            self.img_label_list.append(hm_arr)
    def get_eval(self):
        return self.img_label_list


def collate_ft_fn(batch):
    b_img=[]  # hm
    b_raw=[]  # text desc
    b_proc=[]
    b_act=[]  #label
    b_img_adjacent=[]  # only used during real data fine tuning as additional intra-heatmap contrastive signal

    for normalized_img, text_raw_label, text_proc_label, action_cats, adj_imgs in batch:##action cats may contains multiple variants
        b_img.append(normalized_img)
        b_raw.append(text_raw_label)
        b_proc.append(text_proc_label)
        b_act.append(action_cats)
        b_img_adjacent.append(adj_imgs)

    b_img = np.stack(b_img)
    # b_raw = np.stack(b_raw)
    # b_proc = np.stack(b_proc)
    b_img_adjacent = np.stack(b_img_adjacent)

    # print("Shapes inside finetuning collate fn (orig img, adj img)", b_img.shape, b_img_adjacent.shape)

    return b_img, b_raw, b_proc, b_act, b_img_adjacent##not stack b_act beacuse one activity may have multiple labels for future entention


def collate_fn(batch):
    b_img=[]  # hm
    b_raw=[]  # text desc
    b_proc=[]
    b_act=[]  #label

    for normalized_img, text_raw_label, text_proc_label, action_cats in batch:##action cats may contains multiple variants
        b_img.append(normalized_img)
        b_raw.append(text_raw_label)
        b_proc.append(text_proc_label)
        b_act.append(action_cats)

    b_img = np.stack(b_img)
    # b_raw = np.stack(b_raw)
    # b_proc = np.stack(b_proc)

    return b_img, b_raw, b_proc, b_act##not stack b_act beacuse one activity may have multiple labels for future entention

if __name__=="__main__":
    # exp_folders=["openai_exp01_act000_no001", "openai_exp01_act005_no001"]
    # ds = online_sim_dataset(exp_folders=exp_folders, query_classes=[0, 5], crop_size=(256,224),img_size=(224,224))
    # # dl_train = DataLoader(ds, collate_fn=collate_fn, batch_size=64, shuffle=True, drop_last=True, num_workers=4, prefetch_factor=2)
    # # dl_iter_train=iter(dl_train)
    # # hms, _, _, texts = next(dl_iter_train)
    # # print(texts)
    #
    # fs_train_trial_list = []
    # fs_train_trial_list.extend([
    #     "1709606866_153273",  # 00
    #     "1709607054_765925",  # 01
    #     "1709607205_995125",  # 02
    #     "1709607372_2656453",  # 03
    #     "1709607531_2429712",  # 04
    #     "1709607683_3292575",  # 05
    #     "1709607840_4451723",  # 06
    #     "1709607965_052582",  # 07
    #     "1709608090_995019",  # 08
    #     "1709608214_654744",  # 09
    #     "1709608332_7478955",  # 10
    #     "1709608472_0898292",  # 11
    #     "1709608628_8835044",  # 13
    #     "1709608783_0927992",  # 14
    #     "1709608898_7997708",  # 15
    #     "1709609056_7776437",  # 16
    #     "1709609212_4484148",  # 17
    #     "1709609355_7139902",  # 18
    #     "1709609467_4130135",  # 19
    #     "1709609608_3649662",  # 20
    #     "1709609674_7432973",  # 21
    # ])
    # train_classes_fs = [0, 5]
    # ds_fs_train = real_dataset_21classes(trial_list=fs_train_trial_list, query_classes=train_classes_fs,
    #                                      data_location="./Radar_data_collection/Mar_2024/data_0304/hm_dra_0_100/",
    #                                      crop_size=(256, 224), img_size=(224, 224),
    #                                      ratio=1, order="left_part", sampling_gap=224)
    # ds_train_all = ConcatDataset([ds, ds_fs_train])
    # dl_train_all = DataLoader(ds_train_all, collate_fn=collate_fn, batch_size=64, shuffle=True, drop_last=True, num_workers=4, prefetch_factor=2)
    #
    # dl_iter_train_all=iter(dl_train_all)
    # hms, _, texts, labels = next(dl_iter_train_all)
    # print(texts)
    # print(labels)
    # print("all length:",len(ds_train_all))

    # ds = babel_dataset()
    # dl_train = DataLoader(ds, collate_fn=collate_fn, batch_size=30, shuffle=True,
    #                       drop_last=True, num_workers=1, prefetch_factor=1)
    # dl_iter_train = iter(dl_train)
    # hms, _, texts, _ = next(dl_iter_train)
    # print(texts)

    # ds = babel_dataset_with_image(dataset_list=["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
    #                   'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
    #                   'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"])
    # dl_train = DataLoader(ds, collate_fn=collate_fn, batch_size=16, shuffle=True,
    #                       drop_last=True, num_workers=1, prefetch_factor=1)
    # dl_iter_train = iter(dl_train)
    # i=0
    # while i<=100:
    #     hms, r_imgs, texts, _ = next(dl_iter_train)
    #     print(texts)
    #     i+=1


    #id_sub_act_try
    # exp_ids=[0]
    # sub_ids=[0]
    # act_ids=np.linspace(0,32, 33, )
    # try_ids=[0,1,2]
    # trials_list=[]
    # for exp_id in exp_ids:
    #     for sub_id in sub_ids:
    #         for act_id in act_ids:
    #             for try_id in try_ids:
    #                 trial_name="S{:01d}{:02d}{:02d}{:02d}".format(exp_id, sub_id, int(act_id), try_id)
    #                 trials_list.append(trial_name)
    # trials_list.append("S0002003")
    # ds = local_dataset(data_location="./Radar_data_collection/Mar_2024/data_0505/hm_dra_modified_length_0-100",
    #                    trial_list=trials_list, query_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # # ds=local_dataset(data_location="./Video_data_collection/Mar_2024/4_td_hms/0413_combine_data", trial_list=trials_list, query_classes=[0,1,2,3,4,5,6,7,8,9])
    # dl = DataLoader(ds, collate_fn=collate_fn, batch_size=1, shuffle=True, drop_last=False, num_workers=4, prefetch_factor=2)
    # dl_iter = iter(dl)
    # i=0
    # while i<=1:
    #     hms, _, texts, _ = next(dl_iter)
    #     print(texts)
    #     i+=1

    # id_sub_act_try
    exp_ids = [1]
    sub_ids = [0, 1]
    act_ids = list(range(66))
    act_ids.remove(0)
    act_ids.remove(16)
    act_ids.remove(23)
    act_ids.remove(24)
    act_ids.remove(42)
    act_ids.remove(53)
    # still
    # act_ids.remove(47)
    # act_ids.remove(51)
    try_ids = [0]
    trials_list = []
    for exp_id in exp_ids:
        for sub_id in sub_ids:
            for act_id in act_ids:
                for try_id in try_ids:
                    trial_name = "S{:01d}_{:02d}_{:03d}_{:02d}".format(exp_id, sub_id, int(act_id), try_id)
                    trials_list.append(trial_name)
    ds = local_dataset(data_location="./Radar_data_collection/Mar_2024/data_0505/hm_dra_modified_length_0-100",
                       trial_list=trials_list, query_classes=[2])
    # ds=local_dataset(data_location="./Video_data_collection/Mar_2024/4_td_hms/0413_combine_data", trial_list=trials_list, query_classes=[0,1,2,3,4,5,6,7,8,9])
    dl = DataLoader(ds, collate_fn=collate_fn, batch_size=6, shuffle=True, drop_last=False, num_workers=4, prefetch_factor=2)
    dl_iter = iter(dl)
    i=0
    while i<=1:
        hms, _, texts, labels = next(dl_iter)
        # print(texts)
        print(labels)
        i+=1