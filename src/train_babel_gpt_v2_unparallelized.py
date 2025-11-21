"""
mmCLIP synthetic data pretraining code.

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model_gpt import mmCLIP_gpt_multi_brach_property_v3
from dataset import collate_fn, collate_ft_fn, babel_dataset_gpt, babel_dataset_gpt_alltext, local_dataset, HumanML3DDataset
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
from config_babel_gpt_text import setting_list_babel
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import random
import glob
import csv
import pickle


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

class BCE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.error_metric = nn.BCEWithLogitsLoss()

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
    
text_embs = {}
cat_embs = {}
text_embs_actual = []

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
def gen_label(labels, text_features=None, category_labels=None, weights=None):
    num = len(labels)
    gt_hmtext = np.zeros(shape=(num, num))
    # k is the column and i is the row of the gt matrix here
    for i, label in enumerate(labels):
        match_idx = list(labels).index(label)
        for k in range(num):
            # the hm/text pairs should still be '1' in the ground truth matrix
            if labels[k] == label:
                gt_hmtext[i, k] = 1
            elif text_features is not None:  # used for dynamic similarity matching which doesn't work very well
                gt_hmtext[i, k] = text_features[match_idx, :] @ text_features[k, :]
        # normalization; only has effect when similarity matching is in use
        gt_hmtext[i] = (gt_hmtext[i] - gt_hmtext[i].min()) / (gt_hmtext[i].max() - gt_hmtext[i].min())
    
    if category_labels:
        for (i, cat_labels), cat_weights in zip(enumerate(category_labels), weights):
            for j in range(num):
                labels_in_common = set(category_labels[j]).intersection(set(cat_labels))
                total_weight = 0
                # total weight for this category match is determined by how strong of a match it is
                # use weight corresponding to text here
                for cat, weight in zip(cat_labels, cat_weights):
                    if cat in labels_in_common:
                        total_weight += weight
                if labels_in_common and gt_hmtext[j, i] != 1:
                    assert total_weight != 0, total_weight
                    gt_hmtext[j, i] = total_weight  # soft matching depending on weighting of that category
            
    return gt_hmtext

def gen_category_labels(category_labels, category_labels_used):
    """
    Generate a ground truth matrix for contrastive category alignment loss.

    Each heatmap in the batch will have an associated set of category labels - a positive pair in the ground truth will be
    when the ith heatmap in the matrix has the jth category at position (i, j). 
    """

    gt_cat = np.zeros(shape=(len(category_labels_used),len(category_labels)))

    for i, cat in enumerate(category_labels_used):
        for j, sample_cat_labels in enumerate(category_labels):
            if cat in sample_cat_labels:
                gt_cat[i, j] = 1
    return gt_cat


class KLLoss(nn.Module):
    """
    Used during real data fine-tuning with LoRA. KLD rather than cross entropy is used because the ground-truth distribution
    is no longer a one-hot vector, and so H(P) in D_kl = H(P, Q) - H(P) is no longer zero. In the pretraining case, H(P) is zero
    because a one-hot encoded vector has an entropy of zero, and so cross-entropy (H(P, Q)) is appropriate. Here, H(P) is a positive number.

    CE deemphasizes other large values, while KL divergence does not do this, so it is appropriate in the multiple positive pair case.
    
    Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold. Loss is clamped 
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = nn.KLDivLoss(reduction="batchmean")

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)  # softmaxed logits/probs
        probs2 = F.softmax(label * 10, 1)  # ground truth similarity distributions, why multiplied by 10?
        # softmax between ground truth vectors and softmaxed logits (probabilities)
        loss = self.error_metric(probs1, probs2)  # * batch_size
        return loss

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

np.set_printoptions(suppress=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda:3')  # A5000 on csgpu1. A6000 on csgpu4
print(f"Using device: {device}")

if __name__ == "__main__":
    exp_name = "babel_0505_5set"
    for setting_dict in setting_list_babel:
        exp_setting = setting_dict["exp_setting"]
        print(exp_setting)
        hm_type = setting_dict["hm_type"]

        if not setting_dict["if_use_img"]:
            if setting_dict["model_type"]=="mmCLIP_gpt_multi_head":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"] == "mmCLIP_gpt_multi_brach_property_v3":
                mmclip = mmCLIP_gpt_multi_brach_property_v3(proj_head_dim=64,
                                                         if_use_hm_proj=setting_dict["if_use_hm_proj"],
                                                         if_use_text_proj=setting_dict["if_use_text_proj"],
                                                         if_use_text_att=setting_dict["if_use_text_att"],
                                                         if_use_hm_att=setting_dict["if_use_hm_att"],
                                                         if_use_hm=setting_dict["if_use_hm"],
                                                         device=device,
                                                         in_channels=len(hm_type),
                                                         if_use_hmtext_cross_attn=setting_dict["use_hmtext_cross_attention"]).to(device)
            else:
                assert "Please provide a valid model_type"
        else:
            if setting_dict["model_type"] == "mmCLIP_gpt_multi_head":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property":
                assert "Please provide a valid model_type"
            else:
                assert "Please provide a valid model_type"
        print_trainable_parameters(mmclip)

        use_muon = setting_dict["use_muon"]

        if use_muon:
            adam_params = []
            muon_params = []

            for name, p in mmclip.named_parameters():
                if not p.requires_grad:
                    continue
                if p.dim() == 2:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

            # Adam gradient descent optimizer
            optimizer = torch.optim.Adam([{'params': adam_params, 'lr': setting_dict['lr']}])
            optimizer_muon = torch.optim.Muon(muon_params)  # muon optimizer which ensures momentum updates are full-rank / orthogonal
            # muon also incorporates a weight decay as with AdamW
            scheduler_muon = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_muon, T_max=0.001)
        else:
            optimizer = torch.optim.Adam([{'params': mmclip.parameters(), 'lr': setting_dict['lr']}])

        # Exponential learning rate decay at a rate of 0.9 per epoch to encourage model convergence later in training
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=setting_dict['gamma'])
        loss_KL = KLLoss()  # used during fine-tuning(?)
        loss_ce = CE_Loss()  # main loss used during contrastive training
        loss_bce = BCE_Loss()
        loss_cos = Cos_loss()
        loss_mse = MSE_Loss()
        exp_folders_list = []

        ds= HumanML3DDataset(data_paths="123")
        if setting_dict['if_use_babel']:
            """
            Babel dataset: ta,tr,td heatmaps generated from synthetic signal data from mocap video of human activity, 
            paired with text activity label descriptions
            """
            ds_babel = babel_dataset_gpt(data_paths=setting_dict["babel_train_data_location"],
                                   label_dict_path=setting_dict["label_dict_path"],
                                   dataset_list=setting_dict["dataset_list"],
                                   gpt_data_location=setting_dict["babel_gpt_data_location"],
                                   crop_size=setting_dict["crop_size"],
                                   img_size=setting_dict["img_size"],
                                   if_range_aug=setting_dict['if_range_aug'],
                                   if_use_gpt=setting_dict["if_use_gpt"],
                                   if_use_img=setting_dict["if_use_img"],
                                   use_category_alignment=setting_dict["use_category_alignment"])
            ds=ConcatDataset([ds, ds_babel])
        if setting_dict['if_use_sim_local']:
            # NOT USED DURING PRETRAINING
            """
            Testing dataset consisting of signal heatmaps and corresponding text activity label descriptions
            Unused during training, used as unseen testing data
            """
            train_classes_real = setting_dict["train_classes_real"]
            ds_local = local_dataset(trial_list=setting_dict["trial_list"], query_classes=train_classes_real,
                               data_location=setting_dict["local_train_data_location"],
                               gpt_data_location=setting_dict["gpt_data_location"],
                               crop_size=setting_dict["crop_size"], ratio=setting_dict["train_ratio"],
                               order=setting_dict["train_order"],
                               img_size=setting_dict["img_size"], sampling_gap=setting_dict["train_sampling_gap"],
                               if_range_aug=setting_dict["if_range_aug"])
            ds = ConcatDataset([ds, ds_local])
        if setting_dict["if_use_humanml3d"]:
            """
            HumanML3DDataset (Human Motion-Language 3D Dataset): similarly to babel dataset, return the heatmaps generated from the
            synthetic signal data derived from this dataset, paired with activity label text descriptions created by ChatGPT
            """
            ds_humanml3d = HumanML3DDataset(data_paths=setting_dict["humanml3d_train_data_location"],
                                            text_paths=setting_dict["humanml3d_text_paths"],
                                            gpt_data_location=setting_dict["humanml3d_gpt_location"],
                                            csv_path=setting_dict["humanml3d_cvs_paths"],
                                            dataset_list=setting_dict["dataset_list"],
                                            crop_size=setting_dict["crop_size"],
                                            img_size=setting_dict["img_size"],
                                            if_use_gpt=setting_dict["if_use_gpt"],
                                            if_range_aug=setting_dict["if_range_aug"],
                                            use_category_alignment=setting_dict["use_category_alignment"])
            ds = ConcatDataset([ds, ds_humanml3d])
        if setting_dict["if_use_t2m"]:
            pass

        use_intra_hm = False # setting_dict["num_hm_segs_per_activity"] > 1

        if use_intra_hm:
            dl_train = DataLoader(ds, collate_fn=collate_ft_fn, batch_size=setting_dict["batch_size"], shuffle=True,
                                drop_last=True, num_workers=4, prefetch_factor=2)
        else:
            dl_train = DataLoader(ds, collate_fn=collate_fn, batch_size=setting_dict["batch_size"], shuffle=True,
                                drop_last=True, num_workers=4, prefetch_factor=2)
        dl_iter_train = iter(dl_train)

        test_class_list = setting_dict["test_class_list"]
        ds_dl_val_list = []
        for test_class in test_class_list:
            # Unseen test heatmap/activity label description pairs
            ds_val = local_dataset(trial_list=setting_dict["trial_list"], query_classes=test_class,
                                   data_location=setting_dict["local_test_data_location"],
                                   gpt_data_location=setting_dict["gpt_data_location"],
                                   crop_size=setting_dict["crop_size"], img_size=setting_dict["img_size"],
                                   ratio=setting_dict["test_ratio"], order=setting_dict["test_order"],
                                   sampling_gap=setting_dict["test_sampling_gap"])  # intrahm data turned off by default
            dl_val = DataLoader(ds_val, collate_fn=collate_fn, batch_size=10, shuffle=False, drop_last=False,
                            num_workers=1,
                            prefetch_factor=1)
            ds_dl_val_list.append([ds_val, dl_val])

        def prefetch_text_embs():
            """
            TODO fix ~200 texts not getting embeddings
            Tortured process for prefetching text embeddings and calculating centroids:

            1. load only all text/category pairs from modified babel_dataset_gpt
                ** we cannot just load texts from the original babel_dataset_gpt because the small amount of
                randomness in that dataset (when choosing the segment within the frame to use) causes not all
                texts to be processed
            2. compute text description embeddings and copy into global list text_embs_actual
                ** store map of text descs (text_embs) to their location in text_embs_actual. this is needed
                to prevent storing the embeddings in two places, because we need to...
            3. for each category associated to each text, store a map from the category name to the list of embeddings associated with that category
                ** rather than storing the embeddings themselves, store their locations in text_embs_actual
            4. then compute the centroids for each category using the map from 3. (see gen_category_centroids())
            5. then, during training, we can reuse the precomputed embeddings as well as access the relevant category centroids

            """

            # only prefetch for babel dataset, because only babel dataset has category labels
            # babel_dataset_gpt_alltext only returns the texts for each hm
            ds_babel = babel_dataset_gpt_alltext(data_paths=setting_dict["babel_train_data_location"],
                                   label_dict_path=setting_dict["label_dict_path"],
                                   dataset_list=setting_dict["dataset_list"],
                                   gpt_data_location=setting_dict["babel_gpt_data_location"],
                                   crop_size=setting_dict["crop_size"],
                                   img_size=setting_dict["img_size"],
                                   if_range_aug=setting_dict['if_range_aug'],
                                   if_use_gpt=setting_dict["if_use_gpt"],
                                   if_use_img=setting_dict["if_use_img"],
                                   use_category_alignment=setting_dict["use_category_alignment"])
            global text_embs_actual
            it = 0
            for i in range(len(ds_babel)):
                if i % 100 == 0:
                    print(f"Computing sample {i} / {len(ds_babel)}, computed {it} text embs total")
                sample = ds_babel[i]
                texts = sample[0]
                cats_all = sample[1]
                
                assert len(texts) == len(cats_all)
                for text, cat_frame in zip(texts, cats_all):
                    text_tup = tuple(text)
                    if text_tup in text_embs:
                        continue
                    emb = mmclip.cal_text_features_2d([text])[0].detach().cpu()
                    text_embs_actual.append(emb)
                    text_embs.update({text_tup: it})
                    cat_frame = cat_frame.split(",") if "," in cat_frame else [cat_frame]
                    if len(cat_frame) > 1:  # only use "pure" text embeddings to calculate the category mean embedding,
                        # i.e., text embeddings that only map to that category. this can be removed and experimented with.
                        it += 1
                        continue
                    for cat in cat_frame:
                        if cat not in cat_embs:
                            cat_embs.update({cat: [it]})
                        else:
                            cat_embs[cat].append(it)
                    it += 1
            text_embs_actual = torch.stack(text_embs_actual)


        def gen_category_centroids():
            cat_centroids = {}
            category_size = {}
            for cat in cat_embs.keys():
                embs = torch.index_select(text_embs_actual, 0, torch.tensor(cat_embs[cat]))
                if embs.shape[0] < 5:  # remove category embs with low number of samples, which are likely just noise
                    continue
                embs = embs / embs.norm(dim=-1, keepdim=True)
                cat_centroids.update({cat: []})
                for i in range(0, 6):
                    # calculate centroid along attribute embeddings
                    mean = torch.mean(embs[:, i, :], dim=0)
                    cat_centroids[cat].append(mean / mean.norm(dim=-1, keepdim=True))
                cat_centroids[cat] = torch.stack(cat_centroids[cat])  
                assert cat_centroids[cat].shape == torch.Size([6, 768])   
                category_size.update({cat: embs.shape[0]})                         
            return cat_centroids, category_size

        # Set up log file and model checkpoint files

        iteration_num = setting_dict['iteration_num']
        iteration = 0
        if not os.path.isdir("./src/{}/".format(exp_name)):
            os.mkdir("./src/{}/".format(exp_name))
        if not os.path.isdir("./src/{}/{}/".format(exp_name, exp_setting)):
            os.mkdir("./src/{}/{}/".format(exp_name, exp_setting))

        if not os.path.isdir("./src/{}/{}/confusion_matrix_unpar/".format(exp_name, exp_setting)):
            os.mkdir("./src/{}/{}/confusion_matrix_unpar/".format(exp_name, exp_setting))

        log_file = open("./src/{}/{}/log_unseen_unpar.txt".format(exp_name, exp_setting), "w+")
        for key, value in setting_dict.items():
            log_file.writelines("{}:  {}\n".format(key, value))
        if not os.path.isdir("./src/{}/{}/checkpoint_unseen_unpar".format(exp_name, exp_setting)):
            os.mkdir("./src/{}/{}/checkpoint_unseen_unpar".format(exp_name, exp_setting))
        test_acc_list = []
        avg_test_acc_list=[]
        for i in range(len(ds_dl_val_list)):
            test_acc_list.append([])

        use_cat = setting_dict["use_category_alignment"]

        all_hmtext_loss = []
        if use_cat:
            all_hmcat_loss = []
        if setting_dict["use_hmtext_cross_attention"]:
            all_ca_loss = []

        if use_cat:
            print("\nPrefetching text embeddings and generating category centroids...\n")
            prefetch_text_embs()  # pregenerate text embeddings for all texts in dataset - required to generate centroid category embeddings for category alignment
            # TODO refactor to not use category_size so we can just reuse the pickle file instead of recomputing each time
            cat_centroids, category_size = gen_category_centroids()  # dictionary mapping category names to their centroid embeddings
            pickle.dump(cat_centroids, open("./src/cat_centroids.pkl", "wb"))
            print(len(cat_centroids))

        cat_similarities = {}

        if use_cat:
            centroid_quality_report = open("./src/centroid_report.txt", "w+")
            with torch.no_grad():
                # sanity check that the texts are actually close to the centroids
                ds_babel = babel_dataset_gpt(data_paths=setting_dict["babel_train_data_location"],
                                    label_dict_path=setting_dict["label_dict_path"],
                                    dataset_list=setting_dict["dataset_list"],
                                    gpt_data_location=setting_dict["babel_gpt_data_location"],
                                    crop_size=setting_dict["crop_size"],
                                    img_size=setting_dict["img_size"],
                                    if_range_aug=setting_dict['if_range_aug'],
                                    if_use_gpt=setting_dict["if_use_gpt"],
                                    if_use_img=setting_dict["if_use_img"],
                                    use_category_alignment=setting_dict["use_category_alignment"])
                for i in range(len(ds_babel)):
                    sample = ds_babel[i]
                    texts = [sample[2]]
                    cats = [sample[3]]
                    for text, cats in zip(texts, cats):
                        for cat in cats:
                            if cat not in cat_centroids:
                                continue
                            centroid = cat_centroids[cat]
                            centroid = centroid / centroid.norm(dim=-1, keepdim=True)
                            text_emb = text_embs_actual[text_embs[tuple(text)]]
                            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                            if cat not in cat_similarities:
                                cat_similarities.update({cat: {k: [] for k in range(6)}})
                            for j in range(6):
                                sim = text_emb[j, :] @ centroid[j, :]
                                cat_similarities[cat][j].append(sim)
                for cat in sorted(cat_similarities.keys(), key=lambda cat: category_size[cat]):
                    centroid_quality_report.writelines([f"Centroid {cat} ({category_size[cat]} occurrences):\n"])
                    for i in range(6):
                        sims = cat_similarities[cat][i]
                        mean = np.mean(sims)
                        if i < 5:
                            centroid_quality_report.writelines([
                                f"\tAverage similarity of texts in cat at attribute {i} embedding: {mean}\n",
                                f"\tVariance of texts in cat at {i}: {sum([(x - mean)**2 for x in sims]) / len(sims)}\n"
                            ])
                        else:
                            centroid_quality_report.writelines([
                                f"\tAverage similarity of texts in cat at aggregated embedding: {mean}\n",
                                f"\tVariance of texts in cat at {i}: {sum([(x - mean)**2 for x in sims]) / len(sims)}\n"
                            ])

        def create_tsne(hm_features, labels):
            pca = PCA(n_components=50)
            pca_feats = pca.fit_transform(hm_features)
            tsne = TSNE(n_components=2)
            tsne_feats = tsne.fit_transform(pca_feats)
            assert len(tsne_feats) == len(labels), f"{len(labels)}, {len(tsne_feats)}"

            df = pd.DataFrame({
                'TSNE-1': tsne_feats[:, 0],
                'TSNE-2': tsne_feats[:, 1],
                'label': labels
            })

            # Create the visualization
            plt.figure(figsize=(10, 8))

            # Method 1: Using scatterplot with hue for labels
            sns.scatterplot(
                data=df,
                x='TSNE-1',
                y='TSNE-2',
                hue='label',
                palette='tab10',
                s=50,
                alpha=0.7,
                edgecolor='none'
            )

            plt.title('TSNE Visualization', fontsize=16, fontweight='bold')
            plt.xlabel('TSNE Dimension 1', fontsize=12)
            plt.ylabel('TSNE Dimension 2', fontsize=12)
            plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"./src/babel_0505_5set/tsne_{exp_setting[-1]}_unpar.png")
            plt.clf()

        while iteration <= iteration_num:
            """
            MAIN TRAINING LOOP
            """
            if iteration%1000==1:
                # 1 epoch is 1000 iterations, after each epoch decay the learning rate by 0.9 times
                scheduler.step()
                if use_muon:
                    scheduler_muon.step()

            if iteration != 0 and (iteration % 1000 == 0 or (iteration % 50 == 0 and iteration < 200)):
                # get and log eval metrics five times per epoch
                mmclip.eval()
                top1_list=[]
                for i_ds, (ds_val,dl_val) in enumerate(ds_dl_val_list):
                    top1_correct = 0
                    top2_correct = 0
                    total = 0
                    with torch.no_grad():
                        label_list = []
                        pred_list = []
                        # attribute + aggregated text embeddings of unseen test activity description
                        eval_text_emd = mmclip.cal_text_features_2d(ds_val.inference_description_list)[
                            test_class_list[i_ds]]
                        eval_text_feature = eval_text_emd / eval_text_emd.norm(dim=-1, keepdim=True)  # normalize magnitude of vectors
                        if iteration == iteration_num: 
                            tsne_features_hm = []
                        for i, (hms, _, _, labels) in tqdm(enumerate(dl_val), desc="Computing batch"):
                            eval_hm_array = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)
                            ## get features
                            eval_hm_emd, _ = mmclip.cal_hm_features(eval_hm_array)
                            ## normalize
                            eval_hm_feature = eval_hm_emd / eval_hm_emd.norm(dim=-1, keepdim=True)
                            if iteration == iteration_num:
                                tsne_features_hm.extend(eval_hm_feature[:,-1,:].detach().cpu().numpy())
                            ## get prob and class label
                            logit_scale = mmclip.logit_scale.exp()
                            # inference: dot product between heatmap embedding and candidate text label description embeddings
                            # to get cosine sim matrix
                            logits_per_image = logit_scale * eval_hm_feature[:,-1,:] @ eval_text_feature[:,-1,:].t()
                            # logits_per_image = 100 * eval_hm_feature @ eval_text_feature.t()
                            # after softmax, the the topk probs are the model's predictions
                            probs = logits_per_image.softmax(dim=1).detach().cpu()
                            _, eval_pred_top1 = torch.topk(probs, k=1, dim=1)
                            if len(test_class_list[i_ds]) >= 2:
                                _, eval_pred_top2 = torch.topk(probs, k=2, dim=1)
                            else:
                                eval_pred_top2 = None
                            ## cal acc
                            for j in range(len(logits_per_image)):
                                label_list.append(test_class_list[i_ds][labels[j][0]])
                                pred_list.append(test_class_list[i_ds][eval_pred_top1[j].numpy()[0]])
                                total += 1
                                if any([eval_label in eval_pred_top1[j].numpy() for eval_label in labels[j]]):
                                    top1_correct += 1
                                if eval_pred_top2 != None:
                                    if any([eval_label in eval_pred_top2[j].numpy() for eval_label in labels[j]]):
                                        top2_correct += 1
                                else:
                                    top2_correct = 0
                        if iteration == iteration_num: # iteration_num:
                            create_tsne(tsne_features_hm, label_list)
                        # log metrics after eval
                        cm = confusion_matrix(label_list, pred_list)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_class_list[i_ds])
                        disp.plot()
                        plt.savefig(
                            "./src/{}/{}/confusion_matrix_unpar/{:05d}_cm_{:02d}.png".format(exp_name, exp_setting, iteration, i_ds))
                        plt.close()
                        test_acc_list[i_ds].append(top1_correct / total)
                        acc_quantile = np.quantile(test_acc_list[i_ds], .90)
                        top1_list.append(top1_correct / total)
                        # print("{:02d}th list, top1 acc:{:5f}, top2 acc:{:5f}, 90 quantile acc:{:5f}, max acc:{:5f}"
                        #       .format(i_ds,top1_correct / total, top2_correct / total, acc_quantile, max(test_acc_list[i_ds])))
                        log_file.writelines(
                            "Iteration {}, {:02d}th list, top1 acc:{:5f}, top2 acc:{:5f}, 90 quantile acc:{:5f}, max acc:{:5f} \n"
                            .format(iteration, i_ds, top1_correct / total, top2_correct / total, acc_quantile,
                                    max(test_acc_list[i_ds])))

                avg_test_acc_list.append(np.mean(top1_list))
                avg_acc_quantile = np.quantile(avg_test_acc_list, .90)
                log_file.writelines("Iteration {}, top 1 avg {}, 90 quantile acc:{:5f}, max acc:{:5f} \n".format(iteration, np.mean(top1_list), avg_acc_quantile, max(avg_test_acc_list)))
                log_file.flush()
            if iteration % 15000 == 0 or iteration % 30000 == 0 or iteration % 50000 == 0:
                torch.save(mmclip.state_dict(), "./src/{}/{}/checkpoint_unseen_unpar/{:05d}_checkpoint.pt".format(exp_name, exp_setting, iteration))

            # get next batch from dataloader
            try:
                if use_cat:
                    # the category labels are broader activity labels (i.e., 'walk' for any activity involving walking) that will be used to align
                    # examples of the same category.
                    # because there can be multiple babel category labels assigned to a single segment, we can consider
                    # 1. strong category alignment: if two activities have at least one category in familiar, then align them (put a 1 in the relevant spot in the gt)
                    # 2. weak category alignment: only align if two activities have exactly the same set of categories
                    hms, texts, _, category_labels = next(dl_iter_train)
                else:
                    hms, r_imgs, texts, _ = next(dl_iter_train)
            except StopIteration:
                print("new epoch")
                dl_iter_train = iter(dl_train)
                if use_cat:
                    hms, texts, _, category_labels = next(dl_iter_train)
                else:
                    hms, r_imgs, texts, _ = next(dl_iter_train)
                # scheduler.step()
            """
            Calculate loss on current batch and backpropagate:
            """

            hms = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)  ##
            mmclip.train()
            optimizer.zero_grad()
            if use_muon:
                optimizer_muon.zero_grad()

            hm_emds, _ = mmclip.cal_hm_features(hms)  # agg + attribute embeddings
            """
            if use_cat:
                # since we had to compute the category centroid embeddings, the text embeddings are
                # already computed, so just access them instead of recomputing
                text_emds = []
                for text in texts:
                    if tuple(text) in text_embs:
                        text_emds.append(text_embs_actual[text_embs[tuple(text)]].to(device))  # text_embs[text] is the index of that text's emb
                    else:
                        # humanml3d texts is not be represented in the precomputed embeddings
                        text_emds.append(mmclip.cal_text_features_2d([text])[0])
                text_emds = torch.stack(text_emds).to(device)
            else:
            """
            text_emds = mmclip.cal_text_features_2d(texts)

            if setting_dict["use_hmtext_cross_attention"]:
                # these should be aligned
                # expected shape: [16x56x768]  (batch_size x no. hm tokens x emb_dim)
                hm_token_emds, hm_crossmodal_emds = mmclip.cal_hm_tokens_and_crossmodal_hm_features(texts, hms)

            logit_scale = mmclip.logit_scale.exp()
            if use_cat:
                logit_scale_cat = mmclip.logit_scale_cat.exp()
            if setting_dict["use_hmtext_cross_attention"]:
                logit_scale_ca = mmclip.logit_scale_ca.exp()
            hm_features = hm_emds / hm_emds.norm(dim=-1, keepdim=True)  # normalize magnitude of embeddings
            text_features = text_emds / text_emds.norm(dim=-1, keepdim=True)
            if use_cat:
                category_labels_unique = []  # cannot use set() here because relative order must be preserved
                for labels in category_labels:
                    for label in labels:
                        if label not in category_labels_unique:
                            category_labels_unique.append(label)

                category_labels_used = []
            
                # find unique category centroid embs for the batch
                cat_centroid_features = []
                for i, cat in enumerate(category_labels_unique):
                    if cat in cat_centroids:
                        cat_emb = cat_centroids[cat]
                        category_labels_used.append(cat)
                    else:
                        continue
                    cat_emb = cat_emb / cat_emb.norm(dim=-1, keepdim=True)
                    cat_centroid_features.append(cat_emb)
                cat_centroid_features = torch.stack(cat_centroid_features).to(device)

            # hm_features should also include the attr+agg embeddings for the second 5sec segment from the original heatmap data
            # text_features should also include the attr+agg embeddings for the nearest neighbor text description

            hm_crossattention_loss = 0
            all_loss = 0
            it_hmtext_loss = 0
            it_hmcat_loss = 0

            if setting_dict["use_hmtext_cross_attention"]:
                # 56x56 identity matrix as ground truth for crossmodal attention
                ground_truth_ca = torch.tensor(np.eye(hm_crossmodal_emds.shape[1]), dtype=hm_features.dtype, device=device)

                for j in range(hm_crossmodal_emds.shape[0]):
                    # The hmtext cross attention embedding tensor is of shape (16, 56, 768) and does not (should not) use attribute decomposition.
                    # It's easier conceptually to just iterate over the first dimension and compute the logits for each example in the batch separately, and
                    # then just average them. 

                    # (56x768) @ (768x56) = 56x56 - sim between each hm token and its crossmodal counterpart derived from the text tokens
                    logits_hm_text_ca = logit_scale_ca * hm_crossmodal_emds[j, :, :] @ hm_token_emds[j, :, :].t()         

                    hm_crossattention_loss += loss_KL(logits_hm_text_ca, ground_truth_ca)

            for i in range(hm_features.shape[1]):
                logits_hm_text = logit_scale * hm_features[:, i, :] @ text_features[:, i, :].t()  # dot product between the ith 5 attr embs and aggr emb
                if setting_dict["loss_type"] == "ce":
                    if setting_dict["use_dynamic_similarity_matching"]:
                        text_embs_for_gt = text_features.detach().cpu().numpy()
                        ground_truth =torch.tensor(gen_label(np.array(texts)[:, 0], text_features=text_embs_for_gt[:, i, :]), dtype=hm_features.dtype, device=device)
                        loss_imgs = loss_bce(logits_hm_text, ground_truth)
                        loss_text = loss_bce(logits_hm_text.t(), ground_truth)
                    else:
                        ground_truth = torch.arange(len(hms)).to(device)
                        # Contrastive loss: symmetric CE between 5 attr and aggregated heatmap/text embeddings
                        loss_imgs = loss_ce(logits_hm_text, ground_truth)  # this includes both L_attr and L_cls from paper
                        loss_text = loss_ce(logits_hm_text.t(), ground_truth)
                    total_loss = (loss_imgs + loss_text) / 2  
                    it_hmtext_loss += total_loss
                    if setting_dict["if_use_img"] and i==hm_features.shape[1]-1:
                        r_imgs_embds = mmclip.cal_img_features(r_imgs)
                        r_img_features = r_imgs_embds / r_imgs_embds.norm(dim=-1, keepdim=True)
                        logits_per_hm_img = logit_scale * hm_features[:, i, :] @ r_img_features.t()
                        loss_hm_img = loss_ce(logits_per_hm_img, ground_truth)
                        total_loss+= setting_dict["img_loss_ratio"]*loss_hm_img
                elif setting_dict["loss_type"] == "kl":
                    if setting_dict["use_dynamic_similarity_matching"]:
                        text_embs_for_gt = text_features.detach().cpu().numpy()
                        ground_truth = gen_label(np.array(texts)[:, 0], text_features=text_embs_for_gt[:, i, :])
                        ground_truth = torch.tensor(ground_truth, dtype=hm_features.dtype, device=device)
                    elif use_cat:  # humanml3d dataset does not have category labels
                        # will populate a ground truth with 1s where two activities have at least one of the same category (plus original gt)
                        ground_truth = gen_label(np.array(texts)[:, 0])
                        cat_ground_truth = gen_category_labels(category_labels, category_labels_used)
                        ground_truth = torch.tensor(ground_truth, dtype=hm_features.dtype, device=device)

                        # this ground truth should be [no. of unique category labels in batch]x16
                        cat_ground_truth = torch.tensor(cat_ground_truth, dtype=hm_features.dtype, device=device)

                        # this matmul should be [no. of unique category labels in batch]x768 @ 768x16 = [no. of unique category labels in batch]x16
                        # intuitively, logits_hm_cat is the predicted similarity between each average category emb and each hm emb in the batch
                        logits_hm_cat = logit_scale_cat * cat_centroid_features[:, i, :] @ hm_features[:, i, :].t()
                    else:
                        ground_truth = torch.tensor(gen_label(np.array(texts)[:, 0]), dtype=hm_features.dtype,
                                                        device=device)
                    loss_hm_text = loss_KL(logits_hm_text, ground_truth)
                    if use_cat:
                        loss_hm_cat = loss_KL(logits_hm_cat, cat_ground_truth)
                        it_hmcat_loss += setting_dict["lambda"] * loss_hm_cat
                        it_hmtext_loss += (1 - setting_dict["lambda"]) * loss_hm_text 
                        total_loss = (1 - setting_dict["lambda"]) * loss_hm_text + setting_dict["lambda"] * loss_hm_cat
                    else:
                        it_hmtext_loss += loss_hm_text
                        total_loss = loss_hm_text
                    if setting_dict["if_use_img"] and i==hm_features.shape[1]-1:
                        r_imgs_embds = mmclip.cal_img_features(r_imgs)
                        r_img_features = r_imgs_embds / r_imgs_embds.norm(dim=-1, keepdim=True)
                        logits_per_hm_img = logit_scale * hm_features[:, i, :] @ r_img_features.t()
                        loss_hm_img = loss_KL(logits_per_hm_img, ground_truth)
                        total_loss+= setting_dict["img_loss_ratio"]*loss_hm_img

                elif setting_dict["loss_type"] == "cos":
                    total_loss = loss_cos(text_features[:, i, :], hm_features[:, i, :])
                    if setting_dict["if_use_img"] and i==hm_features.shape[1]-1:
                        r_imgs_embds = mmclip.cal_img_features(r_imgs)
                        r_img_features = r_imgs_embds / r_imgs_embds.norm(dim=-1, keepdim=True)
                        loss_hm_img = loss_cos(r_img_features, hm_features[:, i, :])
                        total_loss+= setting_dict["img_loss_ratio"]*loss_hm_img
                elif setting_dict["loss_type"] == "mse":
                    total_loss = loss_mse(text_emds[:, i, :], hm_emds[:, i, :])
                else:
                    assert "Please provide a valid loss function"
                all_loss += total_loss
            if setting_dict["use_hmtext_cross_attention"]:
                all_loss = (1 - setting_dict["lambda"]) * all_loss + setting_dict["lambda"] * hm_crossattention_loss
            all_loss.backward()  # Backpropagate through the model weights using differentiable loss function
            optimizer.step()  # Update weights using Adam gradient descent optimizer
            if use_muon:
                optimizer_muon.step()

            all_hmtext_loss.append(it_hmtext_loss.cpu().item()/6)
            if use_cat:
                all_hmcat_loss.append(it_hmcat_loss.cpu().item()/6)

            if iteration % 200 == 0:
                # for line in logits_per_image.softmax(dim=1).detach().cpu().numpy():
                #     print(line)
                print("iteration:{}, aggr emb loss:{:5f}".format(iteration, total_loss.item()))
                log_file.writelines("iteration:{}, aggr emb loss:{:5f}\n".format(iteration, total_loss.item()))
                print("iteration:{}, hm/text loss:{:5f}".format(iteration, it_hmtext_loss.cpu().item() / 6))
                log_file.writelines("iteration:{}, hm_text loss:{:5f}\n".format(iteration, it_hmtext_loss.cpu().item() / 6))
                if use_cat:
                    print("iteration:{}, category alignment loss:{:5f}".format(iteration, it_hmcat_loss.cpu().item() / 6))
                    log_file.writelines("iteration:{}, category alignment loss:{:5f}\n".format(iteration, it_hmcat_loss.cpu().item() / 6))
                if setting_dict["use_hmtext_cross_attention"]:
                    print("iteration:{}, token alignment loss:{:5f}".format(iteration, hm_crossattention_loss.cpu().item() / hms.shape[0]))
                    log_file.writelines("iteration:{}, token alignment loss:{:5f}\n".format(iteration, hm_crossattention_loss.cpu().item() / hms.shape[0]))

            if setting_dict["use_hmtext_cross_attention"]:
                all_ca_loss.append(hm_crossattention_loss.cpu().item() / hms.shape[0])
            
            iteration += 1

        i = 0
        for loss in [all_hmtext_loss]:
            loss_avg = list(np.convolve(loss, np.ones(5) / 5, "valid"))
            loss_avg.extend(loss[-4:])
            plt.plot(loss)
            plt.title("Heatmap to Text Contrastive Loss (5it Moving Average)")
            plt.ylabel("Loss")
            plt.xlabel("Iteration")
            plt.savefig(f"./src/babel_0505_5set/loss_plot_hmtext_pt_{i}_unpar.png")
            plt.clf()
            i += 1

