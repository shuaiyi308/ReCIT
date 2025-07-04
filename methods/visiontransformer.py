# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#import pywt
# from utils import trunc_normal_
##### gradient reversal layer #####
from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None
revgrad = GradientReversal.apply

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
###################################
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def tensor2img(img,name):
    from torchvision import utils as vutils
    vutils.save_image(img, name, normalize=True)

def patchify(imgs, p=16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    imgs to x (3,224,224) to (3,196,768)
    """
    h, w = imgs.shape[-2:]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = h // p
    w = w // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x
def reverse_patchify(x, p=16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    x to imgs (3,224,224) to (3,196,768) to (3,224,224)
    """
    n = 224//p
    imgs = x.reshape(shape=(x.shape[0],n*n,p*p,3))
    imgs = imgs.reshape(shape=(x.shape[0],n*n,p,p,3))
    imgs = imgs.reshape(shape=(x.shape[0],n,n,p,p,3))
    imgs = torch.einsum('nhwpqc->nchpwq', imgs) 
    imgs = imgs.reshape(shape=(x.shape[0], 3, n*p, n*p))
    
   
    return imgs




class ReCIT(nn.Module):

    def __init__(self, exclude_first_token=True):
        super().__init__()
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x, imgs ,sim=0.3, sty_std = 1000, epoch=0, use_embed=False, testing=False,keepdim=False,exchange=False,):
        # patchify images sim距离值
        imgs = patchify(imgs)

        if testing:
            return x, None, None

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        N, L, D = x.shape  # (B,196,768)
        learnable_sty =torch.zeros(N, L, L, 1).cuda()
        trunc_normal_(learnable_sty, std = sty_std)
   

        if use_embed:

            x_abs = x.norm(dim=-1)
            Sim_matrix = torch.einsum("aik,ajk->aij", x, x) / torch.einsum("ai,aj->aij", x_abs, x_abs)
            
            mean = imgs.mean(dim=-1, keepdim=True)
            var = imgs.var(dim=-1, keepdim=True)
            imgs = (imgs - mean) / (var + 1.e-6) ** .5
            imgs_abs = imgs.norm(dim=-1)
            Sim_matrix_imgs = torch.einsum("aik,ajk->aij", imgs, imgs) / torch.einsum("ai,aj->aij", imgs_abs, imgs_abs)
            #ratio = 1
            ratio = epoch / 50   # Linear adaption
            Sim_matrix = Sim_matrix * ratio + Sim_matrix_imgs * (1 - ratio)
            
         
        else:
            # Pixel Norm
            mean = imgs.mean(dim=-1, keepdim=True)
            var = imgs.var(dim=-1, keepdim=True)
            imgs = (imgs - mean) / (var + 1.e-6) ** .5
            imgs_abs = imgs.norm(dim=-1)
            # (B,L,L)(B,196,196)
            Sim_matrix = torch.einsum("aik,ajk->aij", imgs, imgs) / torch.einsum("ai,aj->aij", imgs_abs, imgs_abs)
      
            
       
        Sim_chose_mask = torch.where(Sim_matrix > sim, torch.tensor(1).cuda(), torch.tensor(0).cuda())
    
        Sim_chose_mask = torch.triu(Sim_chose_mask)
 
        for i in range(L):
            if torch.sum(Sim_chose_mask)== N * L or i==L-1:
                break
            else:
                mask = torch.where(Sim_chose_mask[:,i,:] == 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()).unsqueeze(1).cuda()
                whole_mask= torch.cat((Sim_chose_mask[:,:i+1,:],mask.repeat(1,L-i-1,1)),dim=1)
                Sim_chose_mask = Sim_chose_mask*whole_mask

       
        con,sty = self.decompose(x) #b,p,d
        con = con.cuda()
        sty = sty.cuda()
        aug_con = con.detach()
        aug_sty = sty.detach() #b.p,d
        aug_con= aug_con.unsqueeze(1).repeat(1,L,1,1)
        aug_sty= aug_sty.unsqueeze(1).repeat(1,L,1,1)#b,p,p,d
        aug_con = Sim_chose_mask.unsqueeze(-1) * aug_con
        aug_sty = Sim_chose_mask.unsqueeze(-1) * aug_sty #b,p,p,d 
        avg_sty_sum = aug_sty.sum(dim=2, keepdim=True) #b.p.1.d
        sty_mask = torch.where(avg_sty_sum > 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()).squeeze(-2)
        avg_con_sum = aug_con.sum(dim=2, keepdim=True)
        avg_sty_num = Sim_chose_mask.unsqueeze(-1).sum(dim=2, keepdim=True)#b.p.1,1
        avg_con_num = Sim_chose_mask.unsqueeze(-1).sum(dim=2, keepdim=True)
        avg_sty_value = torch.div(avg_sty_sum,avg_sty_num.clamp(min=1e-7))#b.p.1,d
        avg_con_value = torch.div(avg_con_sum,avg_con_num.clamp(min=1e-7))
        std_sty_value =  torch.sqrt(torch.div(torch.sum((aug_sty- avg_sty_value *Sim_chose_mask.unsqueeze(-1)).pow(2),dim=2), avg_sty_num.squeeze(2).clamp(min=1e-7)))
   
        
        sty_para = sty_mask.unsqueeze(1) * learnable_sty
        sty_para_norm = torch.div(sty_para,torch.sum(sty_para,dim=-2).unsqueeze(-2).clamp(min=1e-7))

        various_sty = torch.randn(N,L,L,1).cuda()*std_sty_value.unsqueeze(1)+avg_sty_value.squeeze(2).unsqueeze(1)

        various_sty = sty_para_norm * various_sty
        sum_sty_value = torch.sum(various_sty,dim=2)
        x = self.compose(con,sum_sty_value)

        x = torch.cat((cls_tokens, x) , dim=1)
        
        return x

 # --- DFT ---
    def decompose(self,x):
        fft_im=torch.fft.fftn(x, dim=(-1))# dim=(-2,-1)或（-1）
        fft_amp, fft_pha = torch.abs(fft_im), torch.angle(fft_im)
        #fft_im_center= torch.fft.fftshift(fft_im, dim=(-1))# dim=(-2,-1)
        #fft_amp, fft_pha = torch.abs(fft_im_center), torch.angle(fft_im_center)
        return fft_pha, fft_amp

# --- IFT ---
    def compose(self,phase, amp):
        fft_im = amp*torch.exp((1j) * phase)
        #fft_im_center = amp*torch.exp((1j) * phase)
        #fft_im= torch.fft.ifftshift(fft_im_center, dim=(-1))
        fft_im = torch.fft.ifftn(fft_im, dim=(-1)).float()
        return  fft_im 


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        self.attn_map = attn  
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_aug = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def return_attn(self):  # maran
        return self.attn.attn_map  
    
    def forward(self, x, return_attention=False, params=None):

        y, attn = self.attn(self.norm1(x))
    
        if return_attention:
            return attn
        msa=self.drop_path(y)
        x = x + self.drop_path(y)
        mlp = self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
       
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        # --add ReCIT--
        self.recit = ReCIT() 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.gr=GradientReversal()

 
    #---attention map    
    def return_attn_map(self):  
        attention_maps=[]
        for block in self.blocks:
            attn = block.return_attn()
            attention_maps.append(attn)
        return attention_maps
    #--- 
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):  # positional embedding
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape  # batch chanel w h
        x = self.patch_embed(x)  # patch linear embedding  #batch (img/patchsize)^2+1(position) chanel*patchsize*patchsize
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)  # positional embedding
        return self.pos_drop(x)

    def prepare_tokens_cls_withoutpos(self, x, testing=False):
        B, nc, w, h = x.shape  # batch chanel w h
        x = self.patch_embed(x)  # patch linear embedding  #batch (img/patchsize)^2+1(position) chanel*patchsize*patchsize
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos = self.interpolate_pos_encoding(x, w, h)  # 
        return self.pos_drop(x),self.pos_drop(pos)

    def forward(self, x, params=None, search=False):
       
    
        B, C, W, H = x.shape
        imgs = x.detach()
        if self.training:
    
            if params.epoch < params.warmup_endingepoch: 
                import random
                from torchvision.transforms import Resize
                # warm up
                n = random.randint(params.startsize, params.endsize)
                resize_transform = Resize(16 * n)
                x = resize_transform(x)
                x, pos= self.prepare_tokens_cls_withoutpos(x)
                cls, shuttle_x = x[:, :1], x[:, 1:]
                random_permutation0 = torch.randperm(shuttle_x.size(1))
                aug_x_shuttle = shuttle_x[:,random_permutation0,:]
                aug_x = torch.cat((cls, aug_x_shuttle) , dim=1)
              
            else:
                
                x,pos = self.prepare_tokens_cls_withoutpos(x)
                aug_x = self.recit(x, imgs, params.sim, params.sty_std)
            x = aug_x + pos
  

           
         
            
           
           
            
        else:
           
            x = self.prepare_tokens(x)
   
        
  
        for i,blk in enumerate(self.blocks):
            x = blk(x, params=params)   
        x = self.norm(x)
        return x[:, 0]
    
        


    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output





def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x



if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = VisionTransformer(depth=1)
    y = model(x)
    print(y.shape)

