num_hm_segs_per_activity = 1
iteration_num = 5000
use_adjacent_hm = False
use_dynamic_similarity_matching = True
epoch_name = "src/babel_0505_5set/babel-human3d-kl-mbvitv3nosw-3group-seed2024/checkpoint_unseen/50000_checkpoint.pt"

import numpy as np
import pandas as pd

def load_sync_list(infilename):
    sync_list = []
    csv_file = pd.read_csv(infilename, header=0)
    print(infilename)
    for row_index in range(csv_file.shape[0]):
        trail_name = csv_file["trial_name"][row_index]
        sync_list.append(trail_name)
    return sync_list


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

try_ids = [0]
trials_list = []
for exp_id in exp_ids:
    for sub_id in sub_ids:
        for act_id in act_ids:
            for try_id in try_ids:
                trial_name = "S{:01d}_{:02d}_{:03d}_{:02d}".format(exp_id, sub_id, int(act_id), try_id)
                trials_list.append(trial_name)
trials_list.extend(["S1_02_034_00","S1_02_035_00","S1_02_036_00","S1_02_037_00","S1_02_038_00",
                    "S1_02_039_00","S1_02_040_00","S1_02_041_00","S1_02_043_00","S1_02_044_00",
                    "S1_02_045_00","S1_02_046_00","S1_02_047_00","S1_02_048_00","S1_02_050_00",
                    "S1_02_051_00","S1_02_052_00","S1_02_054_00","S1_02_055_00","S1_02_056_00",
                    "S1_02_057_00","S1_02_058_00","S1_02_060_00","S1_02_061_00","S1_02_063_00",
                    "S1_02_064_00","S1_02_065_00",
                    "S1_03_034_00","S1_03_035_00","S1_03_036_00","S1_03_037_00","S1_03_038_00",
                    "S1_03_039_00","S1_03_040_00","S1_03_041_00","S1_03_043_00","S1_03_044_00",
                    "S1_03_045_00","S1_03_046_00","S1_03_047_00","S1_03_048_00","S1_03_050_00",
                    "S1_03_051_00","S1_03_052_00",
                    ])

all_list = act_ids

setting_list = [

# 3 test sets for zero-shot pretrained model + fine-tuning (these runs averaged is the zeroshot mmCLIP result)

    {"exp_setting": "mbvitv3-ft-seen50klnosw-seed2024-26800-TESTSET1",
     "num_ll": 1, "dim_ll": 64,
     "local_train_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
     "local_test_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
     "gpt_data_location": "./data/local_data/gpt_data_0505/",
     ## hyper-paras
     "if_save_model": True, "if_use_hm_proj": False, "if_use_text_proj": False, "if_use_text_att": True,
     'lr': 0.0001,
     "trial_list": trials_list,
     "train_classes_real": list(set(all_list) - set([2, 5, 7, 13, 14, 21, 26, 28, 30, 47])),
     "train_classes_num": False,
     "test_classes_real": [2, 5, 7, 13, 14, 21, 26, 28, 30, 47],
     "batch_size": 16,
     "loss_type": "kl",
     "train_ratio": 1,
     "test_ratio": 1,
     "train_order": "left_part",
     "test_order": "left_part", "iteration_num": iteration_num,  # or 1000
     ## model
     "model_type": "mmCLIP_gpt_multi_brach_property_v3",
     ## pretrain and ft
     "if_use_babel_pretrain": True, "if_freeze_heatmap_encoder": False, "if_lora_ft": True,
     "epoch_name": epoch_name,

     ## babel cotrain
     "if_babel_cotrain": False, "if_use_gpt": True, "aug_ratio": 1,
     "label_dict_path": "./data/babel_data/1_segs_label_dict_framenone=1_cleancomma/",
     "babel_gpt_data_location": "./data/babel_data/6_prompts/",
     "babel_train_data_location": ["./data/babel_data/4_td_hms_0-100_sigma0.15_0.7_0.15"],
     "dataset_list": ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
                      'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
                      'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"],

     ## humanml cotrain
     "if_humanml3d_cotrain": False,
     "humanml3d_train_data_location": ["./data/humanml3d_data/4_td_hms_0-100_sigma_1_0.15_1_1_0.2"],
     "humanml3d_text_paths": "./data/HumanML3D/HumanML3D/texts/",
     "humanml3d_cvs_paths": "./data/humanml3d_data/index.csv",
     ## hm
     "if_use_hm": True, "if_range_aug": True,
     "if_use_hm_att": True,  ##only are useful when using multi-branch
     "if_few_shot": False,
     "if_linear_prob": False,
     "hm_type": [0, 1, 2],
     "num_hm_segs_per_activity": num_hm_segs_per_activity,  # 1
     "use_adjacent_hm": use_adjacent_hm,
     "use_dynamic_similarity_matching": use_dynamic_similarity_matching,
     "train_sampling_gap": 8,
     "test_sampling_gap": 224,
     "crop_size": (256, 224),
     "img_size": (224, 224),
     # only useful in fs
     "fs_sampling_gap": 8,
     "fs_sample": 1,
     "fs_start_index": 0,
     "fs_text_weight": 0.1,
     },

 {"exp_setting": "mbvitv3-ft-seen50klnosw-seed2024-26800-TESTSET2",
  "num_ll": 1, "dim_ll": 64,
  "local_train_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "local_test_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "gpt_data_location": "./data/local_data/gpt_data_0505/",
  ## hyper-paras
  "if_save_model": True, "if_use_hm_proj": False, "if_use_text_proj": False, "if_use_text_att": True,
  'lr': 0.0001,
  "trial_list": trials_list,
  "train_classes_real": list(set(all_list) - set([3, 4, 6, 8, 12, 17, 19, 25, 31, 33])),
  "train_classes_num": False,
  "test_classes_real": [3, 4, 6, 8, 12, 17, 19, 25, 31, 33],
  "batch_size": 16,
  "loss_type": "kl",
  "train_ratio": 1,
  "test_ratio": 1,
  "train_order": "left_part",
  "test_order": "left_part", "iteration_num": iteration_num,
  ## model
  "model_type": "mmCLIP_gpt_multi_brach_property_v3",
  ## pretrain and ft
  "if_use_babel_pretrain": True, "if_freeze_heatmap_encoder": False, "if_lora_ft": True,
  "epoch_name": epoch_name,



  ## babel cotrain
  "if_babel_cotrain": False, "if_use_gpt": True, "aug_ratio": 1,
  "label_dict_path": "./data/babel_data/1_segs_label_dict_framenone=1_cleancomma/",
  "babel_gpt_data_location": "./data/babel_data/6_prompts/",
  "babel_train_data_location": ["./data/babel_data/4_td_hms_0-100_sigma0.15_0.7_0.15"],
  "dataset_list": ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
                   'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
                   'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"],

  ## humanml cotrain
  "if_humanml3d_cotrain": False,
  "humanml3d_train_data_location": ["./data/humanml3d_data/4_td_hms_0-100_sigma_1_0.15_1_1_0.2"],
  "humanml3d_text_paths": "./data/HumanML3D/HumanML3D/texts/",
  "humanml3d_cvs_paths": "./data/humanml3d_data/index.csv",
  ## hm
  "if_use_hm": True, "if_range_aug": True,
  "if_use_hm_att": True,  ##only are useful when using multi-branch
  "if_few_shot": False,
  "if_linear_prob": False,
  "hm_type": [0, 1, 2],
  "num_hm_segs_per_activity": num_hm_segs_per_activity,
  "use_adjacent_hm": use_adjacent_hm,
  "use_dynamic_similarity_matching": use_dynamic_similarity_matching,
  "train_sampling_gap": 8,
  "test_sampling_gap": 224,
  "crop_size": (256, 224),
  "img_size": (224, 224),
  # only useful in fs
  "fs_sampling_gap": 8,
  "fs_sample": 1,
  "fs_start_index": 0,
  "fs_text_weight": 0.1,
  },

 {"exp_setting": "mbvitv3-ft-seen50klnosw-seed2024-26800-TESTSET3",
  "num_ll": 1, "dim_ll": 64,
  "local_train_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "local_test_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "gpt_data_location": "./data/local_data/gpt_data_0505/",
  ## hyper-paras
  "if_save_model": True, "if_use_hm_proj": False, "if_use_text_proj": False, "if_use_text_att": True,
  'lr': 0.0001,
  "trial_list": trials_list,
  "train_classes_real": list(set(all_list) - set([1, 15, 18, 22, 27, 32, 37, 40, 43, 63])),
  "train_classes_num": False,
  "test_classes_real": [1, 15, 18, 22, 27, 32, 37, 40, 43, 63],
  "batch_size": 16,
  "loss_type": "kl",
  "train_ratio": 1,
  "test_ratio": 1,
  "train_order": "left_part",
  "test_order": "left_part", "iteration_num": iteration_num,
  ## model
  "model_type": "mmCLIP_gpt_multi_brach_property_v3",
  ## pretrain and ft
  "if_use_babel_pretrain": True, "if_freeze_heatmap_encoder": False, "if_lora_ft": True,
  "epoch_name": epoch_name,

  ## babel cotrain
  "if_babel_cotrain": False, "if_use_gpt": True, "aug_ratio": 1,
  "label_dict_path": "./data/babel_data/1_segs_label_dict_framenone=1_cleancomma/",
  "babel_gpt_data_location": "./data/babel_data/6_prompts/",
  "babel_train_data_location": ["./data/babel_data/4_td_hms_0-100_sigma0.15_0.7_0.15"],
  "dataset_list": ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
                   'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
                   'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"],

  ## humanml cotrain
  "if_humanml3d_cotrain": False,
  "humanml3d_train_data_location": ["./data/humanml3d_data/4_td_hms_0-100_sigma_1_0.15_1_1_0.2"],
  "humanml3d_text_paths": "./data/HumanML3D/HumanML3D/texts/",
  "humanml3d_cvs_paths": "./data/humanml3d_data/index.csv",
  ## hm
  "if_use_hm": True, "if_range_aug": True,
  "if_use_hm_att": True,  ##only are useful when using multi-branch
  "if_few_shot": False,
  "if_linear_prob": False,
  "hm_type": [0, 1, 2],
  "num_hm_segs_per_activity": num_hm_segs_per_activity,
  "use_adjacent_hm": use_adjacent_hm,
  "use_dynamic_similarity_matching": use_dynamic_similarity_matching,
  "train_sampling_gap": 8,  # similar idea to stride in CNNs?
  "test_sampling_gap": 224,
  "crop_size": (256, 224),
  "img_size": (224, 224),
  # only useful in fs
  "fs_sampling_gap": 8,
  "fs_sample": 1,
  "fs_start_index": 0,
  "fs_text_weight": 0.1,
  }]
"""
 # everything past here is pure finetuning or fewshot 

 {"exp_setting": "mbvitv3-seen50klnosw",
  "num_ll": 1, "dim_ll": 64,
  "local_train_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "local_test_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "gpt_data_location": "./data/local_data/gpt_data_0505/",
  ## hyper-paras
  "if_save_model": True, "if_use_hm_proj": False, "if_use_text_proj": False, "if_use_text_att": True,
  'lr': 0.0001,
  "trial_list": trials_list,
  "train_classes_real": list(set(all_list) - set([2, 5, 7, 13, 14, 21, 26, 28, 30, 47])),
  "train_classes_num": False,
  "test_classes_real": [2, 5, 7, 13, 14, 21, 26, 28, 30, 47],
  "batch_size": 16,
  "loss_type": "kl",
  "train_ratio": 1,
  "test_ratio": 1,
  "train_order": "left_part",
  "test_order": "left_part", "iteration_num": 10000,
  ## model
  "model_type": "mmCLIP_gpt_multi_brach_property_v3",
  ## pretrain and ft
  "if_use_babel_pretrain": False, "if_freeze_heatmap_encoder": False, "if_lora_ft": False,
  "epoch_name": "N/A",

  ## babel cotrain
  "if_babel_cotrain": False, "if_use_gpt": True, "aug_ratio": 1,
  "label_dict_path": "./data/babel_data/1_segs_label_dict_framenone=1_cleancomma/",
  "babel_gpt_data_location": "./data/babel_data/6_prompts/",
  "babel_train_data_location": ["./data/babel_data/4_td_hms_0-100_sigma0.15_0.7_0.15"],
  "dataset_list": ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
                   'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
                   'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"],

  ## humanml cotrain
  "if_humanml3d_cotrain": False,
  "humanml3d_train_data_location": ["./data/humanml3d_data/4_td_hms_0-100_sigma_1_0.15_1_1_0.2"],
  "humanml3d_text_paths": "./data/HumanML3D/HumanML3D/texts/",
  "humanml3d_cvs_paths": "./data/humanml3d_data/index.csv",
  ## hm
  "if_use_hm": True, "if_range_aug": True,
  "if_use_hm_att": True,  ##only are useful when using multi-branch
  "if_few_shot": False,
  "if_linear_prob": False,
  "hm_type": [0, 1, 2],
  "train_sampling_gap": 8,
  "test_sampling_gap": 224,
  "crop_size": (256, 224),
  "img_size": (224, 224),
  # only useful in fs
  "fs_sampling_gap": 8,
  "fs_sample": 1,
  "fs_start_index": 0,
  "fs_text_weight": 0.1,
  },


{"exp_setting": "mbvitv3-seen50klnosw-1",
  "num_ll": 1, "dim_ll": 64,
  "local_train_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "local_test_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "gpt_data_location": "./data/local_data/gpt_data_0505/",
  ## hyper-paras
  "if_save_model": True, "if_use_hm_proj": False, "if_use_text_proj": False, "if_use_text_att": True,
  'lr': 0.0001,
  "trial_list": trials_list,
  "train_classes_real": list(set(all_list) - set([3, 4, 6, 8, 12, 17, 19, 25, 31, 33])),
  "train_classes_num": False,
  "test_classes_real": [3, 4, 6, 8, 12, 17, 19, 25, 31, 33],
  "batch_size": 16,
  "loss_type": "kl",
  "train_ratio": 1,
  "test_ratio": 1,
  "train_order": "left_part",
  "test_order": "left_part", "iteration_num": 10000,
  ## model
  "model_type": "mmCLIP_gpt_multi_brach_property_v3",
  ## pretrain and ft
  "if_use_babel_pretrain": False, "if_freeze_heatmap_encoder": False, "if_lora_ft": False,
  "epoch_name": "N/A",

  ## babel cotrain
  "if_babel_cotrain": False, "if_use_gpt": True, "aug_ratio": 1,
  "label_dict_path": "./data/babel_data/1_segs_label_dict_framenone=1_cleancomma/",
  "babel_gpt_data_location": "./data/babel_data/6_prompts/",
  "babel_train_data_location": ["./data/babel_data/4_td_hms_0-100_sigma0.15_0.7_0.15"],
  "dataset_list": ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
                   'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
                   'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"],

  ## humanml cotrain
  "if_humanml3d_cotrain": False,
  "humanml3d_train_data_location": ["./data/humanml3d_data/4_td_hms_0-100_sigma_1_0.15_1_1_0.2"],
  "humanml3d_text_paths": "./data/HumanML3D/HumanML3D/texts/",
  "humanml3d_cvs_paths": "./data/humanml3d_data/index.csv",
  ## hm
  "if_use_hm": True, "if_range_aug": True,
  "if_use_hm_att": True,  ##only are useful when using multi-branch
  "if_few_shot": False,
  "if_linear_prob": False,
  "hm_type": [0, 1, 2],
  "train_sampling_gap": 8,
  "test_sampling_gap": 224,
  "crop_size": (256, 224),
  "img_size": (224, 224),
  # only useful in fs
  "fs_sampling_gap": 8,
  "fs_sample": 1,
  "fs_start_index": 0,
  "fs_text_weight": 0.1,
  },

{"exp_setting": "mbvitv3-seen50klnosw-2",
  "num_ll": 1, "dim_ll": 64,
  "local_train_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "local_test_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "gpt_data_location": "./data/local_data/gpt_data_0505/",
  ## hyper-paras
  "if_save_model": True, "if_use_hm_proj": False, "if_use_text_proj": False, "if_use_text_att": True,
  'lr': 0.0001,
  "trial_list": trials_list,
  "train_classes_real": list(set(all_list) - set([1, 15, 18, 22, 27, 32, 37, 40, 43, 63])),
  "train_classes_num": False,
  "test_classes_real": [1, 15, 18, 22, 27, 32, 37, 40, 43, 63],
  "batch_size": 16,
  "loss_type": "kl",
  "train_ratio": 1,
  "test_ratio": 1,
  "train_order": "left_part",
  "test_order": "left_part", "iteration_num": 10000,
  ## model
  "model_type": "mmCLIP_gpt_multi_brach_property_v3",
  ## pretrain and ft
  "if_use_babel_pretrain": False, "if_freeze_heatmap_encoder": False, "if_lora_ft": False,
  "epoch_name": "N/A",

  ## babel cotrain
  "if_babel_cotrain": False, "if_use_gpt": True, "aug_ratio": 1,
  "label_dict_path": "./data/babel_data/1_segs_label_dict_framenone=1_cleancomma/",
  "babel_gpt_data_location": "./data/babel_data/6_prompts/",
  "babel_train_data_location": ["./data/babel_data/4_td_hms_0-100_sigma0.15_0.7_0.15"],
  "dataset_list": ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
                   'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
                   'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"],

  ## humanml cotrain
  "if_humanml3d_cotrain": False,
  "humanml3d_train_data_location": ["./data/humanml3d_data/4_td_hms_0-100_sigma_1_0.15_1_1_0.2"],
  "humanml3d_text_paths": "./data/HumanML3D/HumanML3D/texts/",
  "humanml3d_cvs_paths": "./data/humanml3d_data/index.csv",
  ## hm
  "if_use_hm": True, "if_range_aug": True,
  "if_use_hm_att": True,  ##only are useful when using multi-branch
  "if_few_shot": False,
  "if_linear_prob": False,
  "hm_type": [0, 1, 2],
  "train_sampling_gap": 8,
  "test_sampling_gap": 224,
  "crop_size": (256, 224),
  "img_size": (224, 224),
  # only useful in fs
  "fs_sampling_gap": 8,
  "fs_sample": 1,
  "fs_start_index": 0,
  "fs_text_weight": 0.1,
  },






    {
     "exp_setting": "mbvitv3-fewshot-seen50klnosw-seed2024-26800",
     "num_ll": 1, "dim_ll": 64,
     "local_train_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
     "local_test_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
     "gpt_data_location": "./data/local_data/gpt_data_0505/",
     ## hyper-paras
     "if_save_model": True, "if_use_hm_proj": False, "if_use_text_proj": False, "if_use_text_att": True,
     'lr': 0.0001,
     "trial_list": trials_list,
     "train_classes_real": list(set(all_list) - set([2, 5, 7, 13, 14, 21, 26, 28, 30, 47])),
     "train_classes_num": False,
     "test_classes_real": [2, 5, 7, 13, 14, 21, 26, 28, 30, 47],
     "batch_size": 16,
     "loss_type": "kl",
     "train_ratio": 1,
     "test_ratio": 1,
     "train_order": "left_part",
     "test_order": "left_part", "iteration_num":1000,
     ## model
     "model_type": "mmCLIP_gpt_multi_brach_property_v3",
     ## pretrain and ft
     "if_use_babel_pretrain": True, "if_freeze_heatmap_encoder": False, "if_lora_ft": True,
     "epoch_name": "src/babel_0505_5set/babel-human3d-kl-mbvitv3nosw-3group-seed2024/checkpoint_unseen/50000_checkpoint.pt",

     ## babel cotrain
     "if_babel_cotrain": False, "if_use_gpt": True, "aug_ratio": 1,
     "label_dict_path": "./data/babel_data/1_segs_label_dict_framenone=1_cleancomma/",
     "babel_gpt_data_location": "./data/babel_data/6_prompts/",
     "babel_train_data_location": ["./data/babel_data/4_td_hms_0-100_sigma0.15_0.7_0.15"],
     "dataset_list": ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
                      'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
                      'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"],

     ## humanml cotrain
     "if_humanml3d_cotrain": False,
     "humanml3d_train_data_location": ["./data/humanml3d_data/4_td_hms_0-100_sigma_1_0.15_1_1_0.2"],
     "humanml3d_text_paths": "./data/HumanML3D/HumanML3D/texts/",
     "humanml3d_cvs_paths": "./data/humanml3d_data/index.csv",
     ## hm
     "if_use_hm": True, "if_range_aug": True,
     "if_use_hm_att": True,  ##only are useful when using multi-branch
     "if_few_shot": True,
     "if_linear_prob": False,
     "hm_type": [0, 1, 2],
     "train_sampling_gap": 8,
     "test_sampling_gap": 224,
     "crop_size": (256, 224),
     "img_size": (224, 224),
     # only useful in fs
     "fs_sampling_gap": 8,
     "fs_sample": 1,
     "fs_start_index": 0,
     "fs_text_weight": 0.1,
     },

 {
  "exp_setting": "mbvitv3-fewshot-seen50klnosw-seed2024-26800-1",
  "num_ll": 1, "dim_ll": 64,
  "local_train_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "local_test_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "gpt_data_location": "./data/local_data/gpt_data_0505/",
  ## hyper-paras
  "if_save_model": True, "if_use_hm_proj": False, "if_use_text_proj": False, "if_use_text_att": True,
  'lr': 0.0001,
  "trial_list": trials_list,
  "train_classes_real": list(set(all_list) - set([3, 4, 6, 8, 12, 17, 19, 25, 31, 33])),
  "train_classes_num": False,
  "test_classes_real": [3, 4, 6, 8, 12, 17, 19, 25, 31, 33],
  "batch_size": 16,
  "loss_type": "kl",
  "train_ratio": 1,
  "test_ratio": 1,
  "train_order": "left_part",
  "test_order": "left_part", "iteration_num": 1000,
  ## model
  "model_type": "mmCLIP_gpt_multi_brach_property_v3",
  ## pretrain and ft
  "if_use_babel_pretrain": True, "if_freeze_heatmap_encoder": False, "if_lora_ft": True,
  "epoch_name": "src/babel_0505_5set/babel-human3d-kl-mbvitv3nosw-3group-seed2024/checkpoint_unseen/50000_checkpoint.pt",

  ## babel cotrain
  "if_babel_cotrain": False, "if_use_gpt": True, "aug_ratio": 1,
  "label_dict_path": "./data/babel_data/1_segs_label_dict_framenone=1_cleancomma/",
  "babel_gpt_data_location": "./data/babel_data/6_prompts/",
  "babel_train_data_location": ["./data/babel_data/4_td_hms_0-100_sigma0.15_0.7_0.15"],
  "dataset_list": ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
                   'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
                   'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"],

  ## humanml cotrain
  "if_humanml3d_cotrain": False,
  "humanml3d_train_data_location": ["./data/humanml3d_data/4_td_hms_0-100_sigma_1_0.15_1_1_0.2"],
  "humanml3d_text_paths": "./data/HumanML3D/HumanML3D/texts/",
  "humanml3d_cvs_paths": "./data/humanml3d_data/index.csv",
  ## hm
  "if_use_hm": True, "if_range_aug": True,
  "if_use_hm_att": True,  ##only are useful when using multi-branch
  "if_few_shot": True,
  "if_linear_prob": False,
  "hm_type": [0, 1, 2],
  "train_sampling_gap": 8,
  "test_sampling_gap": 224,
  "crop_size": (256, 224),
  "img_size": (224, 224),
  # only useful in fs
  "fs_sampling_gap": 8,
  "fs_sample": 1,
  "fs_start_index": 0,
  "fs_text_weight": 0.1,
 },

 {
  "exp_setting": "mbvitv3-fewshot-seen50klnosw-seed2024-26800-2",
  "num_ll": 1, "dim_ll": 64,
  "local_train_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "local_test_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
  "gpt_data_location": "./data/local_data/gpt_data_0505/",
  ## hyper-paras
  "if_save_model": True, "if_use_hm_proj": False, "if_use_text_proj": False, "if_use_text_att": True,
  'lr': 0.0001,
  "trial_list": trials_list,
  "train_classes_real": list(set(all_list) - set([1, 15, 18, 22, 27, 32, 37, 40, 43, 63])),
  "train_classes_num": False,
  "test_classes_real": [1, 15, 18, 22, 27, 32, 37, 40, 43, 63],
  "batch_size": 16,
  "loss_type": "kl",
  "train_ratio": 1,
  "test_ratio": 1,
  "train_order": "left_part",
  "test_order": "left_part", "iteration_num": 1000,
  ## model
  "model_type": "mmCLIP_gpt_multi_brach_property_v3",
  ## pretrain and ft
  "if_use_babel_pretrain": True, "if_freeze_heatmap_encoder": False, "if_lora_ft": True,
  "epoch_name": "src/babel_0505_5set/babel-human3d-kl-mbvitv3nosw-3group-seed2024/checkpoint_unseen/50000_checkpoint.pt",

  ## babel cotrain
  "if_babel_cotrain": False, "if_use_gpt": True, "aug_ratio": 1,
  "label_dict_path": "./data/babel_data/1_segs_label_dict_framenone=1_cleancomma/",
  "babel_gpt_data_location": "./data/babel_data/6_prompts/",
  "babel_train_data_location": ["./data/babel_data/4_td_hms_0-100_sigma0.15_0.7_0.15"],
  "dataset_list": ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
                   'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
                   'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"],

  ## humanml cotrain
  "if_humanml3d_cotrain": False,
  "humanml3d_train_data_location": ["./data/humanml3d_data/4_td_hms_0-100_sigma_1_0.15_1_1_0.2"],
  "humanml3d_text_paths": "./data/HumanML3D/HumanML3D/texts/",
  "humanml3d_cvs_paths": "./data/humanml3d_data/index.csv",
  ## hm
  "if_use_hm": True, "if_range_aug": True,
  "if_use_hm_att": True,  ##only are useful when using multi-branch
  "if_few_shot": True,
  "if_linear_prob": False,
  "hm_type": [0, 1, 2],
  "train_sampling_gap": 8,
  "test_sampling_gap": 224,
  "crop_size": (256, 224),
  "img_size": (224, 224),
  # only useful in fs
  "fs_sampling_gap": 8,
  "fs_sample": 1,
  "fs_start_index": 0,
  "fs_text_weight": 0.1,
 },


]
"""