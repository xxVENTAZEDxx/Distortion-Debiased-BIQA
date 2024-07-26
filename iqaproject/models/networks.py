import torch
import copy
from torch import nn
import timm
from einops import rearrange
from torch.nn import functional as F

def create_model(model_name):
    if model_name.startswith('resnet'):
        model = timm.create_model(model_name, features_only=True, out_indices=(1, 2, 3), pretrained=True)

    else:
        model = timm.create_model(model_name, pretrained=True)

    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6
    print(f'{model_name} number of parameters: {num_params_m:.2f}M')

    return model


class IQAModel(nn.Module):
    def __init__(self, backbone_name='vit_small_patch16_224', embed_dim=384):
        super(IQAModel, self).__init__()

        self.backbone, self.class_branch = self.build_backbone_classifier(backbone_name=backbone_name)
        
        for name, module in self.backbone._modules.items():
            
            if name.startswith('patch_embed'):
                for param in module.parameters():
                    param.requires_grad = False
               
            if name.startswith('blocks'):
                for i, block in enumerate(module):
                    for param in block.parameters():
                        param.requires_grad = False
                    if i == 4:
                        break

        for param in self.class_branch.parameters():
            param.requires_grad = False
        


        self.patches_resolution = 14
        #################### classification branch ####################

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 2)  # Change the output size to 2
        )
        for param in self.fc.parameters():
            param.requires_grad = False

        #################### single distorted image branch ####################
        selected_blocks = self.build_blocks(backbone_name=backbone_name)
        self.up_branch = selected_blocks

        self.conv1 = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=3, stride=1, padding=1) #nn.Sequential(
            # nn.Conv2d(embed_dim * 4, embed_dim * 4, kernel_size=1, stride=1),
            # nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=3, stride=1, padding=1))
        self.channel_atten1 = Attention(dim=self.patches_resolution ** 2, out_dim=self.patches_resolution ** 2, num_heads=1)
        self.channel_atten2 = Attention(dim=self.patches_resolution ** 2, out_dim=self.patches_resolution ** 2, num_heads=1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        # self.atten = Attention(dim=embed_dim, out_dim=embed_dim)
        #################### restore image branch ####################

        self.bottom_branch = copy.deepcopy(selected_blocks)
        self.block_idx = len(self.bottom_branch) - 3
        self.cross_atten = Attention(dim=embed_dim, out_dim=embed_dim)

        self.score_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1)
        )

        self.score_dist = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1)
        )


    def build_backbone_classifier(self, backbone_name, layer_num=6):
        backbone = create_model(backbone_name)
        selected_blocks = copy.deepcopy(backbone.blocks[layer_num:9])

        del backbone.blocks[layer_num:]
        # for i, block in enumerate(backbone.blocks):
        #     print(f"Remaining block {i}: {block}")
        return backbone, selected_blocks

    def build_blocks(self, backbone_name, layer_num=6, block_num=6):
        backbone = create_model(backbone_name)
        selected_blocks = copy.deepcopy(backbone.blocks[layer_num:layer_num+block_num])

        block_list = nn.ModuleList()
        for block in selected_blocks:
            block_list.append(block)

        del backbone

        return block_list

    def forward_features(self, x, block_list):
        features = []

        # 遍历ModuleList，提取特征并添加到列表中
        for i, block in enumerate(block_list):

            x = block(x)
            if i >= self.block_idx:
                features.append(x[:, 1:])

        features = torch.cat(features, dim=-1)
        features = rearrange(features, 'b (h w) c -> b c (h w)', h=self.patches_resolution, w=self.patches_resolution)

        features = self.channel_atten1(features)
        features = rearrange(features, 'b c (h w) -> b c h w', h=self.patches_resolution, w=self.patches_resolution)
        features = self.conv1(features)
        # orig_features = features
        features = rearrange(features, 'b c h w -> b c (h w)', h=self.patches_resolution, w=self.patches_resolution)

        features = self.channel_atten2(features)
        features = rearrange(features, 'b c (h w) -> b c h w', h=self.patches_resolution, w=self.patches_resolution)
        features = self.conv2(features)
        # features = features + orig_features
        features = rearrange(features, 'b c h w -> b (h w) c', h=self.patches_resolution, w=self.patches_resolution)


        return features

    # def forward(self, dist, restore, ref=None):

        
    #     f_dist = self.backbone.forward_features(dist)
    #     f_dis = self.forward_features(f_dist, self.up_branch)

    #     q_dis = self.score_dist(f_dis.mean(dim=1))
        
    #     return q_dis.flatten()





    def forward(self, dist, restore, ref=None):

        f_res = self.backbone.forward_features(restore)
        f_dist = self.backbone.forward_features(dist)

        ############ classification branch ############
        f_cls = self.class_branch(f_dist)
        weights = self.fc(f_cls.mean(dim=1)).softmax(dim=1)
        # print(weights, weights.shape)
        ###############################################
        diff_res = f_res - f_dist
        diff_res = self.forward_features(diff_res, self.bottom_branch)
        f_dis = self.forward_features(f_dist, self.up_branch)

        f_fusion = self.cross_atten(f_dis, q=diff_res)
        q_res_dis = self.score_fusion(f_fusion.mean(dim=1))
        q_dis = self.score_dist(f_dis.mean(dim=1))

        weighted_q = torch.sum(weights * torch.cat([q_res_dis, q_dis], dim=1), dim=1)

        # if self.training and ref is not None:
        if ref is not None:
            f_ref = self.backbone.forward_features(ref)
            diff_ref = f_ref - f_dist
            diff_ref = self.forward_features(diff_ref,self.bottom_branch)
            error_loss = F.mse_loss(diff_ref, diff_res)
            # f_ref = self.backbone_q.forward_features(ref)
            # error_loss = F.mse_loss(f_res, f_ref)

            return weighted_q, error_loss
        else:
            return weighted_q



class Attention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, q=None):
        B, N, C = x.shape
        if q is None:
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            q = self.q(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



if __name__ == '__main__':
    x = torch.rand(4, 3, 224, 224)
    model = IQAModel()
    print(model(x, x, x))
    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6
    print(f'the number of parameters: {num_params_m:.2f}M')