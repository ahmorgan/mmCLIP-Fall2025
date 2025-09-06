"""
Fine-tune the larger CLIP model for more iterations because the model is larger and needs more training.
Try the pretrained CLIP model checkpoint in the research Google Folder.
Try sentence-T5 and confirm Shuai's results.
"""

"""
Config file for model used during pretraining and fine-tuning.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset
from functools import partial
from PIL import Image
import sys
sys.path.append('.')
from lib.VisionTransformer import VisionTransformer,Tent_ViT, ViT_wo_patch_embed, MB_ViT_v3, MB_ViT_v3_shareweight
from timm.models.vision_transformer import VisionTransformer as timm_vit
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class mmCLIP_gpt_multi_brach_property_v3(nn.Module): ##implemented as paper
    def __init__(self, proj_head_dim=64, if_use_hm_proj=False, if_use_text_proj=False, if_use_hm_att=True,
                 if_use_text_att=True, if_use_hm=True, device=None, in_channels=3):
        """
        Every transformer model used in the mmCLIP system is defined and configured here. This includes:
            -self.heatmap_encoder
                -3 heatmap encoder ViTs for each heatmap type (td, tr, ta)
                -ViT for learning 5 attribute class token embeddings for sequence of heatmap patches
                -uses 56 patch tokens for each heatmap
                -forward() returns the 5 learned attribute class token embeddings
            -self.clip_encoder
                -text encoder for learning 5 attribute class token embeddings
                -forward() returns the embedding of one of the 5 sentences of the activity text descriptions, which each represent a single attribute
            -self.clip_processor (not a model but highly relevant to models; essentially tokenizer for self.heatmap_encoder)
                -CLIP image processor, which is used for tokenizing images into patches for the heatmap encoder ViTs
            -self.hm_self_attention
                -transformer used for heatmap attribute class token aggregation
                -forward() returns the learned aggregated class token for the 5 attribute class tokens
            -self.text_self_attention
                -transformer used for text attribute class token aggregation
                -forward() returns the learned aggregated class token for the 5 attribute class tokens
        """
        super().__init__()

        if if_use_hm:
            # self.heatmap_encoder contains three VisionTransformer models self.td_encoder, self.ta_encoder, self.tr_encoder for each heatmap
            # self.heatmap_encoder.forward() returns the 5 attribute class tokens
            self.heatmap_encoder=MB_ViT_v3()
            # self.heatmap_encoder=MB_ViT_v3_shareweight()
        else:
            assert NotImplementedError

        self.if_use_hm_attn = if_use_hm_att
        self.if_use_text_attn = if_use_text_att
        if self.if_use_hm_attn:
            # Transformer used during heatmap attribute embedding aggregation
            self.hm_self_attention = ViT_wo_patch_embed(global_pool=False, embed_dim=128*3, depth=1,
                                                  num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                  norm_layer=partial(nn.LayerNorm, eps=1e-6))  # in: B*L*C
            self.hm_attn_proj = nn.Sequential(nn.Linear(128*3, 768))  # linearly project hm attribute embeddings to embedding dimension of text attribtue embeddings
        
        if self.if_use_text_attn:
            # Transformer used during text attribute embedding aggregation
            # 512 is embedding dimension of the CLIP text encoder
            # ViT can also be used here because ViT is just a standard transformer architecture but called "ViT" when used with image patches
            self.text_self_attention = ViT_wo_patch_embed(global_pool=False, embed_dim=768, depth=1,
                                                  num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                  norm_layer=partial(nn.LayerNorm, eps=1e-6))  # in: B*L*C
            # self.text_attn_proj = nn.Sequential(nn.Linear(512, 512))  # Linear projection doing all the heavy lifting here
        
        self.if_use_hm_proj = if_use_hm_proj
        self.if_use_text_proj = if_use_text_proj
        # unsure what the below linear projections are for but they appear to be unused across the board
        if if_use_hm_proj:
            self.hm_proj = nn.Sequential(nn.Linear(512, proj_head_dim))
        if if_use_text_proj:
            self.text_proj = nn.Sequential(nn.Linear(512,proj_head_dim))
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # temperature scaling in CLIP loss function (?)
        self.clip_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").requires_grad_(False)  # smallest CLIP model
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")  # openai/clip-vit-large-patch14

        # parameters in CLIP text encoder are frozen
        for param in self.clip_encoder.parameters():
            param.requires_grad = False
        self.device = device

    def cal_text_features_2d(self, text_list_2d):  # B*k  # forward() method for CLIP text encoder + self.text_self_attention model
        """
        Generates and returns the 5 attribute text embeddings and aggregated text embedding using CLIP text encoder

        text_list_2d: list of sentences from activity label description generated using ChatGPT
        """

        length = len(text_list_2d)
        text_branches = len(text_list_2d[0])

        text_list = list(np.array(text_list_2d).reshape(-1))
        # Return padded (space is added or truncated to meet input length of 77) PyTorch tensors for each of the 5 attribute embeddings
        text_input = self.clip_processor(text=text_list, return_tensors="pt", padding=True).to(self.device)
        # get 5 text attribute embeddings (one for each sentence from ChatGPT)
        text_embeds = self.clip_encoder.get_text_features(**text_input)
        text_embeds = (text_embeds.reshape(length, text_branches, -1))

        if self.if_use_text_attn:
            # get aggregated text embedding 
            text_embeds_att, _ = self.text_self_attention(text_embeds)
        else:
            text_embeds_att = text_embeds.mean(dim=1)

        text_embeds = torch.cat([text_embeds, text_embeds_att.unsqueeze(1)], dim=1)
        # text_embeds=self.text_attn_proj(text_embeds)

        if self.if_use_text_proj:
            text_embeds = self.text_proj(text_embeds)

        return text_embeds

    def cal_hm_features(self, hm_input):
        """
        Generates and returns the aggregated and 5 attribute heatmap embeddings.

        hm_input: 3-dimensional tensor: a list of 3 (one for each heatmap type) lists of flattened image patches (each patch is itself a list of real numbers).
        """

        # generate individual 128-dim embeddings from each heatmap encoder (td, ta, tr) and return 5 attribute class tokens
        hm_embeds = self.heatmap_encoder(hm_input)
        # return aggregated class token for 5 attribute embeddings
        out, _=self.hm_self_attention(hm_embeds)
        result=torch.cat([hm_embeds, out.unsqueeze(1)], dim=1)  # batch_size * 6 * 384
        # linearly project to embedding dimension of text embeddings
        hm_embeds=self.hm_attn_proj(result)

        if self.if_use_hm_proj:
            hm_embeds = self.hm_proj(hm_embeds)
        
        return hm_embeds, None


    def nearest_neighbor(self, x, batch=None):
        # nearest neighbor of input x in its corresponding frozen feature bank
        # computed for only the text modality right now
        # given a text description embedding, find its nearest neighbor in the batch or frozen feature bank
        # maximize the cosine similarity between the relevant heatmap embedding and both the matching and NN text emebdding
        
        idx = np.argsort(np.array(batch @ x).reshape(-1))[-2:][0]  # get index of nearest neighbor (not itself)
        return batch[idx]


class TextEncoder(nn.Module):
    """ Used by Tent implementation """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND, RoPE?
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD, RoPE?
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    """
    Used in Tent to build prompts for creating activity label text descriptions
    """

    def __init__(self, train_classnames, test_classnames, clip_model):
        super().__init__()

        classnames=train_classnames+test_classnames

        self.train_n_cls=len(train_classnames)
        self.test_n_cls = len(test_classnames)
        n_cls = len(classnames)
        n_ctx = 32#cfg.TRAINER.COOP.N_CTX
        ctx_init = False#cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if False:#cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]## N class prompt
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        self.train_tokenized_prompts=tokenized_prompts[:self.train_n_cls]
        self.test_tokenized_prompts=tokenized_prompts[self.train_n_cls:]

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)#N*words*dim



        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "middle"#cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def get_train_prompt(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.train_n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.train_n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.train_n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def get_test_prompt(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.test_n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.train_n_cls, self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i-self.train_n_cls: i-self.train_n_cls + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i-self.train_n_cls: i-self.train_n_cls + 1, half_n_ctx:, :]



                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.train_n_cls, self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def forward(self):
        train_prompts=self.get_train_prompt()
        test_prompts=self.get_test_prompt()

        return train_prompts, test_prompts

class Tent(nn.Module):
    def __init__(self, train_classnames, test_classnames, proj_head_dim=64, if_use_hm_proj=False, if_use_text_proj=False, if_use_text_att=False, if_use_hm=True, device=None, in_channels=3):
        super().__init__()
        clip_model, self.clip_processor = clip.load("ViT-B/32", device="cpu")  ##utilize clip github implementation
        for param in clip_model.parameters():
            param.requires_grad = False

        self.prompt_learner = PromptLearner(train_classnames, test_classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.train_tokenized_prompts=self.prompt_learner.train_tokenized_prompts
        self.test_tokenized_prompts=self.prompt_learner.test_tokenized_prompts

        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        if if_use_hm:
            self.heatmap_encoder = VisionTransformer(global_pool=False, img_size=(224, 224), in_chans=in_channels,
                                                     patch_size=(224, 4), embed_dim=512, depth=6,
                                                     num_heads=8, mlp_ratio=4, qkv_bias=True,
                                                     norm_layer=partial(nn.LayerNorm, eps=1e-6))

            # self.heatmap_encoder=Tent_ViT()
        else:
            assert NotImplementedError
        self.if_use_hm_proj = if_use_hm_proj
        if if_use_hm_proj:
            # self.hm_proj = nn.Linear(512, 512)
            # self.hm_proj = nn.Sequential(nn.Linear(512,256), nn.Linear(256,proj_head_dim))
            self.hm_proj = nn.Sequential(nn.Linear(512,proj_head_dim))
    def cal_hm_features(self, hm_input):
        hm_embeds = self.heatmap_encoder(hm_input).squeeze(1)
        # hm_embeds=self.pc_encoder(hm_input)
        if self.if_use_hm_proj:
            hm_embeds = self.hm_proj(hm_embeds)
        return hm_embeds, None

    def get_train_text_features(self):
        train_prompts,_ = self.prompt_learner()
        tokenized_prompts = self.train_tokenized_prompts
        text_features = self.text_encoder(train_prompts, tokenized_prompts)
        return text_features

    def get_test_text_features(self):
        _, test_prompts = self.prompt_learner()
        tokenized_prompts = self.test_tokenized_prompts
        text_features = self.text_encoder(test_prompts, tokenized_prompts)
        return text_features

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits




if __name__=="__main__":
    tent=Tent(train_classnames=["walk", "run"], test_classnames=["a person sit", "he jump"])

