from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        # x = self.fc_norm(x)
        # x = self.head_drop(x)
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        # x = self.forward_head(x)
        return x


class ViT_wo_patch_embed(timm.models.vision_transformer.VisionTransformer):
    """
    Attribute aggregation module transformer
    """

    def __init__(self, global_pool=False, **kwargs):
        super(ViT_wo_patch_embed, self).__init__(**kwargs)
        self.global_pool = global_pool

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        embed_dim=kwargs['embed_dim']
        self.pos_embed = nn.Parameter(torch.randn(1, 6, embed_dim) * .02)##adjustable
    def forward_features(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:  # Determine the contents of self.block (how it is instantiated)
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        return x, None




class MB_ViT_v3(timm.models.vision_transformer.VisionTransformer):
    """
    Heatmap patch encoder layer + attribute class token embedding learner
    """

    def __init__(self):
        super().__init__(class_token=False, global_pool='avg',reg_tokens=5,
                             no_embed_class=True, embed_dim=384, depth=6,#4
                             num_heads=8, mlp_ratio=4, qkv_bias=True,#4
                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # timm is a library for working with image models (cnns, vit, etc.) built on top of PyTorch.
        # patch_size is 224*4 pixels and images are grayscale (in_chans=1), so there are 56 patch tokens per heatmap
        # Should the image size for each heatmap encoder be different size? (not all 224,224)?
        self.td_encoder=timm.models.vision_transformer.VisionTransformer(class_token=False, global_pool='avg',
                                                                         img_size=(224, 224), in_chans=1,
                                                                         patch_size=(224, 4), embed_dim=128, depth=4,#2
                                                                         num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                                         norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                                                         )
        self.tr_encoder=timm.models.vision_transformer.VisionTransformer(class_token=False, global_pool='avg',
                                                                         img_size=(224, 224), in_chans=1,
                                                                         patch_size=(224, 4), embed_dim=128, depth=4,
                                                                         num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                                         norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                                                         )
        self.ta_encoder=timm.models.vision_transformer.VisionTransformer(class_token=False, global_pool='avg',
                                                                         img_size=(224, 224), in_chans=1,
                                                                         patch_size=(224, 4), embed_dim=128, depth=4,
                                                                         num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                                         norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                                                         )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        embed_dim = 384
        self.reg_tokens=5
        self.norm=norm_layer(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, int(224/4), int(3*128)) * .02)

    def forward(self, x: torch.Tensor):
        # Heatmap encoder
        td_emd = self.td_encoder.forward_features(x[:, 0:1, ...])#->B*L*d
        tr_emd = self.tr_encoder.forward_features(x[:, 1:2, ...])
        ta_emd = self.ta_encoder.forward_features(x[:, 2:3, ...])
        # Concat heatmap encodings
        x = torch.cat([td_emd, tr_emd, ta_emd], dim=2)#->B*L*3d
        x = self._pos_embed(x)#->B*(L+5)*3d
        x = self.patch_drop(x)  # remove certain neurons in mlp to prevent overfitting Read more into this***

        # Attribute token learner
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)## added norm here
        return x[:, 0:self.reg_tokens]  # return five attribute embeddings (5 class tokens)

class MB_ViT_v3_shareweight(timm.models.vision_transformer.VisionTransformer):
    def __init__(self):
        super().__init__(class_token=False, global_pool='avg',reg_tokens=5,
                             no_embed_class=True, embed_dim=384, depth=6,#4
                             num_heads=8, mlp_ratio=4, qkv_bias=True,#4
                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.tdtrta_encoder=timm.models.vision_transformer.VisionTransformer(class_token=False, global_pool='avg',
                                                                         img_size=(224, 224), in_chans=1,
                                                                         patch_size=(224, 4), embed_dim=128, depth=4,#2
                                                                         num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                                         norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                                                         )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        embed_dim = 384
        self.reg_tokens=5
        self.norm=norm_layer(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, int(224/4), int(3*128)) * .02)

    def forward(self, x: torch.Tensor):
        td_emd = self.tdtrta_encoder.forward_features(x[:, 0:1, ...])#->B*L*d
        tr_emd = self.tdtrta_encoder.forward_features(x[:, 1:2, ...])
        ta_emd = self.tdtrta_encoder.forward_features(x[:, 2:3, ...])
        x = torch.cat([td_emd, tr_emd, ta_emd], dim=2)#->B*L*3d
        x = self._pos_embed(x)#->B*(L+5)*3d
        x = self.patch_drop(x)


        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)## added norm here
        return x[:, 0:self.reg_tokens]



class Tent_ViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self):
        super().__init__(class_token=False, global_pool='avg',reg_tokens=5,
                             no_embed_class=True, embed_dim=384, depth=6,#4
                             num_heads=8, mlp_ratio=4, qkv_bias=True,#4
                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.td_encoder=timm.models.vision_transformer.VisionTransformer(class_token=False, global_pool='avg',
                                                                         img_size=(224, 224), in_chans=1,
                                                                         patch_size=(224, 4), embed_dim=128, depth=4,#2
                                                                         num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                                         norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                                                         )
        self.tr_encoder=timm.models.vision_transformer.VisionTransformer(class_token=False, global_pool='avg',
                                                                         img_size=(224, 224), in_chans=1,
                                                                         patch_size=(224, 4), embed_dim=128, depth=4,
                                                                         num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                                         norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                                                         )
        self.ta_encoder=timm.models.vision_transformer.VisionTransformer(class_token=False, global_pool='avg',
                                                                         img_size=(224, 224), in_chans=1,
                                                                         patch_size=(224, 4), embed_dim=128, depth=4,
                                                                         num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                                         norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                                                         )



        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        embed_dim = 384
        self.reg_tokens=1
        self.norm=norm_layer(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, int(224/4), int(3*128)) * .02)
        self.proj=nn.Linear(384,512)
    def forward(self, x: torch.Tensor):
        td_emd = self.td_encoder.forward_features(x[:, 0:1, ...])#->B*L*d
        tr_emd = self.tr_encoder.forward_features(x[:, 1:2, ...])
        ta_emd = self.ta_encoder.forward_features(x[:, 2:3, ...])
        x = torch.cat([td_emd, tr_emd, ta_emd], dim=2)#->B*L*3d
        x = self._pos_embed(x)#->B*(L+5)*3d
        x = self.patch_drop(x)
        for blk in self.blocks:
            x = blk(x)
        out=x[:, 0:self.reg_tokens]#only use the first token
        result=self.proj(out)
        return result




if __name__ == '__main__':
    heatmap_encoder = MB_ViT_v2()
    data=torch.randn((16, 3, 224, 224))
    out=heatmap_encoder(data)
    print(out.shape)


    # model = VisionTransformer(img_size=(256,384),in_chans=1,
    #     patch_size=(256,4), embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6)).cuda(0)
    # img = torch.randn(2, 1, 256, 384).cuda(0)
    # preds = model(img)  # (1, 1000)
    # print(preds.shape)