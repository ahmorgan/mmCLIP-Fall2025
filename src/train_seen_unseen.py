import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model_gpt import mmCLIP_gpt_multi_brach_property_v3
from dataset import collate_fn, collate_ft_fn, babel_dataset_gpt, local_dataset, local_dataset_fs, HumanML3DDataset
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
import cv2
from tqdm import tqdm
# from config_zs_real_signal_seen import setting_list
from config_zs_real_ft import setting_list
from torch.nn import functional as F
from peft import LoftQConfig, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, ConcatDataset, Dataset

import random
import os
seed=42#2024#42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class CE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.error_metric = nn.CrossEntropyLoss()
    def forward(self, prediction, label):
        loss = self.error_metric(prediction, label)
        return loss
class Cos_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.error_metric = nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, prediction, text_feature):
        cos = self.error_metric(prediction, text_feature)
        cosine_loss = (1 - cos).mean()
        return cosine_loss
class MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.error_metric = nn.MSELoss()
    def forward(self, prediction, label):
        loss = self.error_metric(prediction, label)
        return loss
class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = nn.KLDivLoss(reduction="batchmean")

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) #* batch_size
        return loss

def gen_label(labels, text_features=None):
    """
    Given a list of activity labels (texts associated with the hms in a batch), convert it to a one-hot encoded matrix of label column vectors.

    Since we are using KL-divergence during pretraining, the ground truth vector may havem multiple positive pairs.

    This function has been altered to include dynamic similarity matching, i.e. the ground truth similarity matrix

         hm  hm  hm
    text CS  1   CS
    text 1   CS  CS 
    text CS  CS  1    etc.

    where CS for every entry is the cosime similarity of the text matched with the hm and the jth text
    """
    num = len(labels)
    gt = np.zeros(shape=(num, num))
    # k is the column and i is the row of the gt matrix here
    for i, label in enumerate(labels):
        match_idx = list(labels).index(label)
        for k in range(num):
            # the hm/text pairs should still be '1' in the ground truth matrix
            if labels[k] == label:
                gt[i, k] = 1
            elif text_features is not None:
                # text feats should be batch_size * 6 * emb_dim - using aggregated embedding for cosine sim here
                gt[i, k] = text_features[match_idx, :] @ text_features[k, :]  # cosine sim of label i and hm k; embeddings will have already been normalized, so cs is just a dot product
        gt[i] =  (gt[i] - gt[i].min()) / (gt[i].max() - gt[i].min())  # normalize them

    return gt

np.set_printoptions(suppress=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda:0')

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            # print(name)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
if __name__=="__main__":
    exp_name = "zero-shot-mmCLIP-all"
    for setting_dict in setting_list:
        exp_setting=setting_dict["exp_setting"]
        print(exp_setting)
        if setting_dict["if_use_hm"]:
            hm_type=setting_dict["hm_type"]
            if setting_dict["model_type"]=="mmCLIP_gpt_v2":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_branch":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_head":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property_v3":
                mmclip = mmCLIP_gpt_multi_brach_property_v3(proj_head_dim=64, if_use_hm_proj=setting_dict["if_use_hm_proj"],
                                       if_use_text_proj=setting_dict["if_use_text_proj"],
                                       if_use_text_att=setting_dict["if_use_text_att"],
                                       if_use_hm_att=setting_dict["if_use_hm_att"],
                                       if_use_hm=setting_dict["if_use_hm"],
                                       device=device,
                                       in_channels=len(hm_type)).to(device)
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property_with_image":
                assert "Please provide a valid model_type"
            else:
                assert "Please provide a valid model_type"

        else:
            assert "Please provide a valid model_type"

        # if setting_dict["if_use_babel_pretrain"]:
        #     babel_checkpoint_path=setting_dict["epoch_name"]
        #     model_keys=torch.load(babel_checkpoint_path)
        #     for key in list(model_keys.keys()):
        #         model_keys[key.replace('heatmap_encoder.', '')] = model_keys.pop(key)
        #     mmclip.heatmap_encoder.load_state_dict(model_keys, strict=False)
        #     # mmclip.logit_scale.load_state_dict(model_keys, strict=False)
        if setting_dict["if_use_babel_pretrain"]:
            babel_checkpoint_path=setting_dict["epoch_name"]
            model_keys=torch.load(babel_checkpoint_path)
            # mmclip.load_state_dict(model_keys, strict=True)
            mmclip.load_state_dict(model_keys, strict=False)

        if setting_dict["if_freeze_heatmap_encoder"]:
            for param in mmclip.heatmap_encoder.parameters():
                param.requires_grad = False
        # print_trainable_parameters(mmclip)

        if setting_dict["if_lora_ft"]:
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["qkv"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
            )
            if setting_dict["model_type"]=="mmCLIP_gpt_multi_head":#this get_peft_model worl in-place
                lora_model = get_peft_model(mmclip.heatmap_encoder5, config)#previous only use this
                lora_model = get_peft_model(mmclip.heatmap_encoder1, config)
                lora_model = get_peft_model(mmclip.heatmap_encoder2, config)
                lora_model = get_peft_model(mmclip.heatmap_encoder3, config)
                lora_model = get_peft_model(mmclip.heatmap_encoder4, config)
            elif (setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property"
                  or setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property_with_image"):
                lora_model = get_peft_model(mmclip.heatmap_encoder, config)#do not need a return
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property_v3":
                lora_model = get_peft_model(mmclip.heatmap_encoder, config)#do not need a return
                lora_model = get_peft_model(mmclip.hm_self_attention, config)
                lora_model = get_peft_model(mmclip.text_self_attention, config)
                # for param in mmclip.text_self_attention.parameters():
                #     param.requires_grad = False
                # mmclip= get_peft_model(mmclip, config)#not include logitscale, hm_self_attn
            else:
                raise NotImplementedError
            # print_trainable_parameters(lora_model)
        optimizer = torch.optim.Adam([{'params': mmclip.parameters(), 'lr':setting_dict["lr"]}])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        loss_KL = KLLoss()
        loss_img=CE_Loss()
        loss_cos=Cos_loss()
        loss_mse=MSE_Loss()
        exp_folders_list=[]

        print_trainable_parameters(mmclip)

        if setting_dict["if_use_hm"]:
            train_classes_real = setting_dict["train_classes_real"]
            if setting_dict["train_classes_num"]:
                train_classes_real=random.sample(train_classes_real, setting_dict["train_classes_num"])

            setting_dict["used_train_classes_real"]=train_classes_real
            print("used_train_classes_real", train_classes_real)

            # local_dataset (real fine-tuning data) is the only dataset used
            ds = local_dataset(trial_list=setting_dict["trial_list"], query_classes=train_classes_real,
                              data_location=setting_dict["local_train_data_location"],
                               gpt_data_location=setting_dict["gpt_data_location"],
                              crop_size=setting_dict["crop_size"],ratio=setting_dict["train_ratio"], order=setting_dict["train_order"],
                               img_size=setting_dict["img_size"], sampling_gap=setting_dict["train_sampling_gap"],
                               num_hm_segs_per_activity=setting_dict["num_hm_segs_per_activity"], use_adjacent_hm=setting_dict["use_adjacent_hm"])
            if setting_dict["if_babel_cotrain"]:
                # ds_babel = babel_dataset(data_paths=setting_dict["babel_train_data_location"],
                #                    dataset_list=setting_dict["dataset_list"], crop_size=setting_dict["crop_size"],
                #                    img_size=setting_dict["img_size"])
                ds_babel = babel_dataset_gpt(data_paths=setting_dict["babel_train_data_location"],
                                       label_dict_path=setting_dict["label_dict_path"],
                                       dataset_list=setting_dict["dataset_list"],
                                       gpt_data_location=setting_dict["babel_gpt_data_location"],
                                       crop_size=setting_dict["crop_size"],
                                       img_size=setting_dict["img_size"],
                                       if_range_aug=setting_dict["if_range_aug"],
                                       if_use_gpt=setting_dict["if_use_gpt"],
                                       aug_ratio=setting_dict["aug_ratio"])
                ds = ConcatDataset([ds, ds_babel])
            if setting_dict["if_humanml3d_cotrain"]:
                ds_humanml3d=HumanML3DDataset(data_paths=setting_dict["humanml3d_train_data_location"],
                                              text_paths=setting_dict["humanml3d_text_paths"],
                                              csv_path=setting_dict["humanml3d_cvs_paths"],
                                              dataset_list=setting_dict["dataset_list"],
                                              crop_size=setting_dict["crop_size"],
                                              img_size=setting_dict["img_size"], aug_ratio=setting_dict["aug_ratio"],
                                              if_range_aug=setting_dict["if_range_aug"])
                ds = ConcatDataset([ds, ds_humanml3d])

            use_intra_hm = setting_dict["num_hm_segs_per_activity"] > 1  # whether or not to compute and optimize contrastive loss between adjacent heatmaps

            if use_intra_hm:
                dl_train = DataLoader(ds, collate_fn=collate_ft_fn, batch_size=setting_dict["batch_size"], shuffle=True, drop_last=True, num_workers=1, prefetch_factor=4)
            else:
                dl_train = DataLoader(ds, collate_fn=collate_fn, batch_size=setting_dict["batch_size"], shuffle=True, drop_last=True, num_workers=4, prefetch_factor=2)
            dl_iter_train = iter(dl_train)

            test_classes_real = setting_dict["test_classes_real"]
            print("used_test_classes_real", test_classes_real)
            # validation set should use default num_hm_segs_per_activity value of 1
            ds_val = local_dataset(trial_list=setting_dict["trial_list"], query_classes=test_classes_real,
                                  data_location=setting_dict["local_test_data_location"],
                                   gpt_data_location=setting_dict["gpt_data_location"],
                                   crop_size=setting_dict["crop_size"],img_size=setting_dict["img_size"],
                                   ratio=setting_dict["test_ratio"], order=setting_dict["test_order"], sampling_gap=setting_dict["test_sampling_gap"])
            dl_val = DataLoader(ds_val, collate_fn=collate_fn, batch_size=10, shuffle=False, drop_last=False, num_workers=4, prefetch_factor=2)

            if setting_dict["if_few_shot"] or setting_dict["if_linear_prob"]:
                ds_fs=local_dataset_fs(trial_list=setting_dict["trial_list"], query_classes=test_classes_real,
                                  data_location=setting_dict["local_test_data_location"],
                                   gpt_data_location=setting_dict["gpt_data_location"],
                                   crop_size=setting_dict["crop_size"],img_size=setting_dict["img_size"],
                                   ratio=setting_dict["test_ratio"], order=setting_dict["test_order"], sampling_gap=setting_dict["fs_sampling_gap"],
                                start_index=setting_dict["fs_start_index"], few_shot_sample=setting_dict["fs_sample"])
                fs_hm_list=ds_fs.get_eval()


        else:
            train_classes_real = setting_dict["train_classes_real"]
            ds = local_dataset_pc(trial_list=setting_dict["trial_list"], query_classes=train_classes_real,
                               data_location=setting_dict["local_train_data_location"],
                               gpt_data_location=setting_dict["gpt_data_location"],
                               pc_crop_size=setting_dict["pc_crop_size"],ratio=setting_dict["train_ratio"], order=setting_dict["train_order"],
                               sampling_gap=setting_dict["train_sampling_gap"])
            dl_train = DataLoader(ds, collate_fn=collate_fn, batch_size=setting_dict["batch_size"], shuffle=True,
                                  drop_last=True, num_workers=4, prefetch_factor=2)
            dl_iter_train = iter(dl_train)

            test_classes_real = setting_dict["test_classes_real"]
            ds_val = local_dataset_pc(trial_list=setting_dict["trial_list"], query_classes=test_classes_real,
                                   data_location=setting_dict["local_test_data_location"],
                                   gpt_data_location=setting_dict["gpt_data_location"],
                                   pc_crop_size=setting_dict["pc_crop_size"],
                                   ratio=setting_dict["test_ratio"], order=setting_dict["test_order"], sampling_gap=setting_dict["test_sampling_gap"])
            dl_val = DataLoader(ds_val, collate_fn=collate_fn, batch_size=10, shuffle=False, drop_last=False,
                                num_workers=4, prefetch_factor=2)
            if setting_dict["if_few_shot"]:
                assert NotImplementedError

        iteration_num = setting_dict["iteration_num"]#10000
        iteration = 0
        if not os.path.isdir("./src/{}/".format(exp_name)):
            os.mkdir("./src/{}/".format(exp_name))
        if not os.path.isdir("./src/{}/{}/".format(exp_name, exp_setting)):
            os.mkdir("./src/{}/{}/".format(exp_name, exp_setting))
        if not os.path.isdir("./src/{}/{}/confusion_matrix/".format(exp_name, exp_setting)):
            os.mkdir("./src/{}/{}/confusion_matrix/".format(exp_name, exp_setting))
        log_file = open("./src/{}/{}/log_unseen.txt".format(exp_name, exp_setting), "w+")
        for key, value in setting_dict.items():
            log_file.writelines("{}:  {}\n".format(key, value))
        if not os.path.isdir("./src/{}/{}/checkpoint_unseen".format(exp_name, exp_setting)):
            os.mkdir("./src/{}/{}/checkpoint_unseen".format(exp_name, exp_setting))
        test_acc_list=[]
        all_intrahm_loss = []
        all_hmtext_loss = []
        while iteration <= iteration_num:
            if iteration % 200 == 0 or (iteration%50==0 and iteration<600):
                mmclip.eval()
                top1_correct = 0
                top2_correct = 0
                total = 0
                with torch.no_grad():
                    label_list = []
                    pred_list = []
                    eval_text_emd = mmclip.cal_text_features_2d(ds_val.inference_description_list)[test_classes_real]
                    eval_text_feature = eval_text_emd / eval_text_emd.norm(dim=-1, keepdim=True)
                    if setting_dict["if_few_shot"]:
                        fs_hm_emds = []
                        if setting_dict["if_use_hm"]:
                            for i, hms in enumerate(fs_hm_list):
                                fs_hm_array = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)
                                fs_hm_emd, scores = mmclip.cal_hm_features(fs_hm_array)
                                fs_hm_emds.append(torch.mean(fs_hm_emd, dim=0))
                        else:
                            # fs_hm_emds_list=[]
                            # label_list=[]
                            # for i, hms in enumerate(fs_hm_list):
                            #     fs_hm_array = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)
                            #     fs_hm_emd, scores = mmclip.cal_hm_features(fs_hm_array)
                            #     fs_hm_emd=fs_hm_emd[:,-1,:]
                            #     labels=[[i]*len(fs_hm_emds)]
                            #     fs_hm_emds_list.append(fs_hm_emd)
                            #     label_list.append(labels)
                            # classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
                            # classifier.fit(fs_hm_emd, train_labels)
                            assert NotImplementedError
                        fs_hm_emds = torch.stack(fs_hm_emds)
                    elif setting_dict["if_linear_prob"]:
                        assert "linear prob not implemented yet"
                    tsne_hm_list=[]
                    tsne_label_list=[]
                    for i, (hms, _, _, labels) in tqdm(enumerate(dl_val), desc="Computing batch"):
                        if setting_dict["if_use_hm"]:
                            eval_hm_array = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)
                        else:
                            eval_hm_array = torch.from_numpy(hms).float().to(device)
                        ## get features
                        eval_hm_emd, scores = mmclip.cal_hm_features(eval_hm_array)
                        ## normalize
                        eval_hm_feature = eval_hm_emd / eval_hm_emd.norm(dim=-1, keepdim=True)
                        ## get prob and class label
                        logit_scale = mmclip.logit_scale.exp()
                        logits_hm_text = logit_scale * eval_hm_feature[:,-1,:] @ eval_text_feature[:,-1,:].t()
                        probs = logits_hm_text.softmax(dim=1).detach().cpu()
                        if setting_dict["if_few_shot"]:
                            fs_hm_emds = fs_hm_emds / fs_hm_emds.norm(dim=-1, keepdim=True)
                            logits_hm_fs = logit_scale * eval_hm_feature[:,-1,:] @ fs_hm_emds[:,-1,:].t()
                            probs_hm_fs = logits_hm_fs.softmax(dim=1).detach().cpu()
                            probs = setting_dict["fs_text_weight"] * probs + (
                                        1 - setting_dict["fs_text_weight"]) * probs_hm_fs
                        if setting_dict["if_linear_prob"]:
                            assert "linear prob not implemented yet"
                        _, eval_pred_top1 = torch.topk(probs, k=1, dim=1)
                        if len(setting_dict["test_classes_real"])>=2:
                            _, eval_pred_top2 = torch.topk(probs, k=2, dim=1)
                        else:
                            eval_pred_top2=None
                        ## cal acc
                        for j in range(len(logits_hm_text)):
                            label_list.append(test_classes_real[labels[j][0]])
                            pred_list.append(test_classes_real[eval_pred_top1[j].numpy()[0]])
                            total += 1
                            if any([eval_label in eval_pred_top1[j].numpy() for eval_label in labels[j]]):
                                top1_correct += 1
                            if eval_pred_top2!=None:
                                if any([eval_label in eval_pred_top2[j].numpy() for eval_label in labels[j]]):
                                    top2_correct += 1
                            else:
                                top2_correct=0
                    cm = confusion_matrix(label_list, pred_list)
                    # cm=cm.astype('float')/cm.astype('float').sum(axis=1)[:,np.newaxis]
                    # np.savetxt("./src/{}/{}/confusion_matrix/{:05d}_cm.csv".
                    #            format(exp_name, exp_setting, iteration)
                    #            , cm, delimiter=",",fmt="%.2f")
                    plt.clf()
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_classes_real)
                    disp.plot()
                    plt.savefig(
                        "./src/{}/{}/confusion_matrix/{:05d}_cm_ft.png".format(exp_name, exp_setting, iteration))
                    plt.close()


                test_acc_list.append(top1_correct / total)
                acc_quantile=np.quantile(test_acc_list, .90)
                print("top1 acc:{:5f}, top2 acc:{:5f}, 90 quantile acc:{:5f}, max acc:{:5f}"
                      .format(top1_correct / total,top2_correct / total, acc_quantile, max(test_acc_list)))
                log_file.writelines(
                    "Iteration {}, top1 acc:{:5f}, top2 acc:{:5f}, 90 quantile acc:{:5f}, max acc:{:5f} \n"
                    .format(iteration, top1_correct / total,top2_correct / total, acc_quantile, max(test_acc_list)))
                log_file.flush()
            if setting_dict["if_save_model"] and iteration % 1000 == 0:
                torch.save(mmclip.heatmap_encoder.state_dict(),
                        "./src/{}/{}/checkpoint_unseen/{:05d}_checkpoint_ft.pt"
                        .format(exp_name, exp_setting, iteration))
                            
            try:
                if use_intra_hm:
                    hms, _, texts, _, hms_adj = next(dl_iter_train)  # we need to optimize contrastive loss between hms and hms_adj (hms_adjacent) as well
                    if hms_adj.shape[1] == 1:
                        hms_adj = np.reshape(hms_adj, newshape=(hms_adj.shape[0], hms_adj.shape[2], hms_adj.shape[3], hms_adj.shape[4]))
                    else:
                        raise NotImplementedError
                else:
                    hms, _, texts, _ = next(dl_iter_train)
            except StopIteration:
                print("new epoch")
                log_file.writelines("new epoch\n")
                dl_iter_train = iter(dl_train)
                if use_intra_hm:
                    hms, _, texts, _, hms_adj = next(dl_iter_train)
                    if hms_adj.shape[1] == 1:
                        hms_adj = np.reshape(hms_adj, newshape=(hms_adj.shape[0], hms_adj.shape[2], hms_adj.shape[3], hms_adj.shape[4]))
                    else:
                        raise NotImplementedError
                else:
                    hms, _, texts, _ = next(dl_iter_train)
                scheduler.step()
            if setting_dict["if_use_hm"]:
                hms = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)##
                if use_intra_hm:
                    hms_adj = torch.from_numpy(hms_adj[:, hm_type, ...]).float().to(device)
            else:
                hms = torch.from_numpy(hms).float().to(device)
            mmclip.train()
            optimizer.zero_grad()

            hm_emds, scores = mmclip.cal_hm_features(hms)
            if use_intra_hm:
                hm_adj_emds, scores = mmclip.cal_hm_features(hms_adj)
            
            text_emds = mmclip.cal_text_features_2d(texts)
            logit_scale = mmclip.logit_scale.exp()
            hm_features = hm_emds / hm_emds.norm(dim=-1, keepdim=True)
            if use_intra_hm:
                hm_adj_features = hm_adj_emds / hm_adj_emds.norm(dim=-1, keepdim=True)
            text_features = text_emds / text_emds.norm(dim=-1, keepdim=True)

            all_loss = 0
            it_intrahm_loss = 0
            it_hmtext_loss = 0
            for i in range(hm_features.shape[1]):
                # this iterates over the 6 embeddings for each example in the batch: 5 attr + 1 aggregated
                logits_hm_text = logit_scale * hm_features[:,i,:] @ text_features[:,i,:].t()  # compute cosine sim for each 16 embs against other 16 embs in correspoding modality
                if use_intra_hm:
                    logits_intra_hm = []
                    logits_intra_hm.append(logit_scale * hm_features[:, i, :] @ hm_adj_features[:, i, :].t())  # heatmaps from the same activity should be similar
                if setting_dict["loss_type"] == "ce":
                    ground_truth = torch.arange(len(hms)).to(device)
                    loss_imgs = loss_img(logits_hm_text, ground_truth)
                    loss_text=loss_img(logits_hm_text.t(), ground_truth)
                    if use_intra_hm:
                        # this loss should also be symmetric, right?
                        # contrastive loss between two semantically similar heatmaps
                        loss_intra_hm = loss_img(sim_matrix, ground_truth)
                        loss_intra_hm += loss_img(sim_matrix.t(), ground_truth)
                        loss_intra_hm /= 2
                        total_loss = (loss_imgs+loss_text+loss_intra_hm)/3
                    else:
                        total_loss = (loss_imgs+loss_text)/2
                elif setting_dict["loss_type"] == "kl":
                    text_embs_for_gt = None
                    if setting_dict["use_dynamic_similarity_matching"]:
                        text_embs_for_gt = text_features.detach().cpu().numpy()
                        ground_truth = torch.tensor(gen_label(np.array(texts)[:, 0], text_features=text_embs_for_gt[:, i, :]), dtype=hm_features.dtype,
                                                        device=device)
                    else:
                        ground_truth = torch.tensor(gen_label(np.array(texts)[:, 0]), dtype=hm_features.dtype,
                                                        device=device)
                    loss_hm_text = loss_KL(logits_hm_text, ground_truth)
                    it_hmtext_loss += loss_hm_text
                    if use_intra_hm:
                        loss_intra_hm = 0
                        for sim_matrix in logits_intra_hm:
                            loss_intra_hm += loss_KL(sim_matrix, ground_truth)  # you can use the same ground truth here because hms and hms_adj will always have the same label
                        loss_intra_hm /= len(logits_intra_hm)
                        total_loss = (loss_hm_text+loss_intra_hm)/2
                        it_intrahm_loss += loss_intra_hm
                    else:
                        total_loss = loss_hm_text
                elif setting_dict["loss_type"] == "cos":
                    total_loss=loss_cos(text_features[:,i,:], hm_features[:,i,:])
                elif setting_dict["loss_type"] == "mse":
                    total_loss = loss_mse(text_emds[:,i,:], hm_emds[:,i,:])
                else:
                    assert "Please provide a valid loss function"
                all_loss+=total_loss
            all_loss.backward()
            optimizer.step()

            # will be used to plot the loss over iterations
            if use_intra_hm:
                all_intrahm_loss.append(it_intrahm_loss.cpu().item()/6)
            all_hmtext_loss.append(it_hmtext_loss.cpu().item()/6)

            if iteration % 200 == 0:
                # for line in logits_per_image.softmax(dim=1).detach().cpu().numpy():
                #     print(line)
                print("iteration:{}, loss:{:5f}".format(iteration, total_loss.item()))
                log_file.writelines("iteration:{}, loss:{:5f}\n".format(iteration, total_loss.item()))
                log_file.flush()
            iteration += 1

        # outside of the training loop now
        if use_intra_hm:
            all_intrahm_avg = list(np.convolve(all_intrahm_loss, np.ones(5) / 5, "valid"))  # clever trick using convolution to take a moving average
            all_intrahm_avg.extend(all_intrahm_loss[-4:])  # we can't take the 5-window moving average at the last four elements, just fill them in with the original values
            plt.plot(all_intrahm_avg)
            plt.title("Intra-Heatmap Contrastive Loss (5it Moving Average)")
            plt.ylabel("Loss")
            plt.xlabel("Iteration")
            plt.savefig("./src/zero-shot-mmCLIP-all/" + exp_setting + "/loss_plot_intrahm.png")
            plt.clf()
        all_hmtext_avg = list(np.convolve(all_hmtext_loss, np.ones(5) / 5, "valid"))
        all_hmtext_avg.extend(all_hmtext_loss[-4:])
        plt.plot(all_hmtext_avg)
        plt.title("Heatmap to Text Contrastive Loss (5it Moving Average)")
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        plt.savefig("./src/zero-shot-mmCLIP-all/" + exp_setting + "/loss_plot_hmtext.png")
        plt.clf()

