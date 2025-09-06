import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

sys.path.append(".")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model_gpt import Tent
from dataset import collate_fn, local_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
import cv2
from tqdm import tqdm
from config_tent import setting_list
from torch.nn import functional as F
from peft import LoftQConfig, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, ConcatDataset, Dataset


import random
import os

seed=2024
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
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

np.set_printoptions(suppress=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda:0')


if __name__ == "__main__":
    exp_name = "zero-shot-tent-all-2024"
    for setting_dict in setting_list:
        exp_setting = setting_dict["exp_setting"]
        print(exp_setting)

        if setting_dict["if_use_hm"]:
            hm_type = setting_dict["hm_type"]
            train_classes_real = setting_dict["train_classes_real"]
            if setting_dict["train_classes_num"]:
                train_classes_real = random.sample(train_classes_real, setting_dict["train_classes_num"])

            setting_dict["used_train_classes_real"] = train_classes_real
            print("used_train_classes_real", train_classes_real)

            ds = local_dataset(trial_list=setting_dict["trial_list"], query_classes=train_classes_real,
                               data_location=setting_dict["local_train_data_location"],
                               gpt_data_location=setting_dict["gpt_data_location"],
                               crop_size=setting_dict["crop_size"], ratio=setting_dict["train_ratio"],
                               order=setting_dict["train_order"],
                               img_size=setting_dict["img_size"], sampling_gap=setting_dict["train_sampling_gap"])

            dl_train = DataLoader(ds, collate_fn=collate_fn, batch_size=setting_dict["batch_size"], shuffle=True,
                                  drop_last=True, num_workers=4, prefetch_factor=2)
            dl_iter_train = iter(dl_train)

            test_classes_real = setting_dict["test_classes_real"]
            print("used_test_classes_real", test_classes_real)
            ds_val = local_dataset(trial_list=setting_dict["trial_list"], query_classes=test_classes_real,
                                   data_location=setting_dict["local_test_data_location"],
                                   gpt_data_location=setting_dict["gpt_data_location"],
                                   crop_size=setting_dict["crop_size"], img_size=setting_dict["img_size"],
                                   ratio=setting_dict["test_ratio"], order=setting_dict["test_order"],
                                   sampling_gap=setting_dict["test_sampling_gap"])
            dl_val = DataLoader(ds_val, collate_fn=collate_fn, batch_size=10, shuffle=False, drop_last=False,
                                num_workers=4, prefetch_factor=2)
        else:
            raise NotImplementedError


        train_des = []
        test_des = []
        for i in train_classes_real:
            train_des.append(ds.description_list[i][1])
        for i in test_classes_real:
            test_des.append(ds_val.description_list[i][1])
        tent = Tent(train_classnames=train_des, test_classnames=test_des).to(device)
        print_trainable_parameters(tent)

        optimizer = torch.optim.Adam([{'params': tent.parameters(), 'lr': setting_dict["lr"]}])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        loss_img = CE_Loss()



        iteration_num = setting_dict["iteration_num"]  # 10000
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
        test_acc_list = []
        while iteration <= iteration_num:
            if iteration % 200 == 0 or (iteration % 50 == 0 and iteration < 600):
                tent.eval()
                top1_correct = 0
                top2_correct = 0
                total = 0
                with torch.no_grad():
                    label_list = []
                    pred_list = []
                    eval_text_emd = tent.get_test_text_features()
                    eval_text_feature = eval_text_emd / eval_text_emd.norm(dim=-1, keepdim=True)
                    for i, (hms, _, _, labels) in tqdm(enumerate(dl_val), desc="Computing batch"):
                        if setting_dict["if_use_hm"]:
                            eval_hm_array = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)
                        else:
                            raise NotImplementedError
                        eval_hm_emd, scores = tent.cal_hm_features(eval_hm_array)
                        eval_hm_feature = eval_hm_emd / eval_hm_emd.norm(dim=-1, keepdim=True)
                        ## get prob and class label
                        logit_scale = tent.logit_scale.exp()

                        logits_hm_text = logit_scale * eval_hm_feature[:, :] @ eval_text_feature[:, :].t()
                        probs = logits_hm_text.softmax(dim=1).detach().cpu()
                        _, eval_pred_top1 = torch.topk(probs, k=1, dim=1)
                        if len(setting_dict["test_classes_real"]) >= 2:
                            _, eval_pred_top2 = torch.topk(probs, k=2, dim=1)
                        else:
                            eval_pred_top2 = None
                        ## cal acc
                        for j in range(len(logits_hm_text)):
                            label_list.append(test_classes_real[labels[j][0]])
                            pred_list.append(test_classes_real[eval_pred_top1[j].numpy()[0]])
                            total += 1
                            if any([eval_label in eval_pred_top1[j].numpy() for eval_label in labels[j]]):
                                top1_correct += 1
                            if eval_pred_top2 != None:
                                if any([eval_label in eval_pred_top2[j].numpy() for eval_label in labels[j]]):
                                    top2_correct += 1
                            else:
                                top2_correct = 0
                    cm = confusion_matrix(label_list, pred_list)
                    # cm=cm.astype('float')/cm.astype('float').sum(axis=1)[:,np.newaxis]
                    # np.savetxt("./src/{}/{}/confusion_matrix/{:05d}_cm.csv".
                    #            format(exp_name, exp_setting, iteration)
                    #            , cm, delimiter=",",fmt="%.2f")
                    plt.clf()
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_classes_real)
                    disp.plot()
                    plt.savefig(
                        "./src/{}/{}/confusion_matrix/{:05d}_cm.png".format(exp_name, exp_setting, iteration))
                    plt.close()
                test_acc_list.append(top1_correct / total)
                acc_quantile = np.quantile(test_acc_list, .90)
                print("top1 acc:{:5f}, top2 acc:{:5f}, 90 quantile acc:{:5f}, max acc:{:5f}"
                      .format(top1_correct / total, top2_correct / total, acc_quantile, max(test_acc_list)))
                log_file.writelines(
                    "Iteration {}, top1 acc:{:5f}, top2 acc:{:5f}, 90 quantile acc:{:5f}, max acc:{:5f} \n"
                    .format(iteration, top1_correct / total, top2_correct / total, acc_quantile, max(test_acc_list)))
                log_file.flush()
                if setting_dict["if_save_model"]:
                    torch.save(tent.heatmap_encoder.state_dict(),
                               "./src/{}/{}/checkpoint_unseen/{:05d}_checkpoint.pt"
                               .format(exp_name, exp_setting, iteration))
            try:
                hms, _, _, labels = next(dl_iter_train)
            except StopIteration:
                print("new epoch")
                log_file.writelines("new epoch\n")
                dl_iter_train = iter(dl_train)
                hms, _, _, labels = next(dl_iter_train)
                scheduler.step()
            if setting_dict["if_use_hm"]:
                hms = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)  ##
            else:
                hms = torch.from_numpy(hms).float().to(device)
            tent.train()
            optimizer.zero_grad()

            hm_emds, scores = tent.cal_hm_features(hms)
            text_emds = tent.get_train_text_features()
            logit_scale = tent.logit_scale.exp()
            hm_features = hm_emds / hm_emds.norm(dim=-1, keepdim=True)
            text_features = text_emds / text_emds.norm(dim=-1, keepdim=True)

            logits_hm_text = logit_scale * hm_features[:, :] @ text_features[:, :].t()
            if setting_dict["loss_type"] == "ce":
                ground_truth = torch.tensor(labels, dtype=torch.long).squeeze(1).to(device)
                loss_imgs = loss_img(logits_hm_text, ground_truth)
                total_loss = loss_imgs
            else:
                assert "Please provide a valid loss function"

            total_loss.backward()
            optimizer.step()

            if iteration % 200 == 0:
                # for line in logits_per_image.softmax(dim=1).detach().cpu().numpy():
                #     print(line)
                print("iteration:{}, loss:{:5f}".format(iteration, total_loss.item()))
                log_file.writelines("iteration:{}, loss:{:5f}\n".format(iteration, total_loss.item()))
                log_file.flush()
            iteration += 1