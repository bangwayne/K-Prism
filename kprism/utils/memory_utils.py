import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


# @torch.jit.script
#
# class UpsampleSupport(nn.Module):
#     def __init__(self, feat_channel):
#         super().__init__()
#         self.res4_proj = nn.Conv2d(feat_channel, feat_channel//2, kernel_size=1)
#         self.res3_proj = nn.Conv2d(feat_channel//2, feat_channel//4, kernel_size=1)
#
#     def forward(self, align_data):
#         out = {}
#         out["res5"] = align_data  # original
#
#         # upsample to 64x64 and project channels
#         x_res4 = F.interpolate(align_data, scale_factor=2, mode='bilinear', align_corners=False)
#         x_res4 = self.res4_proj(x_res4)  # [B, 192, 64, 64]
#         out["res4"] = x_res4
#
#         # upsample to 128x128 and project channels
#         x_res3 = F.interpolate(x_res4, scale_factor=2, mode='bilinear', align_corners=False)
#         x_res3 = self.res3_proj(x_res3)  # [B, 96, 128, 128]
#         out["res3"] = x_res3
#
#         return out

def get_similarity(mk: torch.Tensor,
                   qk: torch.Tensor,
                   add_batch_dim: bool = False,
                   rank: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x h * w    - Memory keys
    # qk: B x CK x h * w - Query keys
    # Dimensions in [] are flattened
    # mk.shape:torch.Size([8, 384, 1, 32, 32])
    # qk.shape:torch.Size([8, 384, 32, 32])
    if add_batch_dim:
        mk = mk.unsqueeze(0)
        qk = qk.unsqueeze(0)

    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    # mk.shape:torch.Size([8, 384, 1024T])
    qk = qk.flatten(start_dim=2)
    # [8, 384, 1024]

    a_sq = mk.pow(2).sum(1).unsqueeze(2)
    two_ab = 2 * (mk.transpose(1, 2) @ qk)
    similarity = (-a_sq + two_ab)
    similarity = similarity / math.sqrt(CK)  # B*N*HW
    # print(f"similarity: {similarity.shape}")
    # (B, T×HW, HW)
    if not rank:
        return similarity
    else:
        B, T_HW, HW = similarity.shape  # mk: B x CK x (T * H * W)
        T = T_HW // (HW)
        # similarity: B x (T*HW) x HW
        rank_sim = similarity.view(B, T, HW, HW)  # B x T x HW x HW

        #
        frame_sim = rank_sim.mean(dim=(2, 3))  # B x T
        best_match_idx = frame_sim.argmax(dim=1)
        # print(f"frame_sim: {frame_sim}")
        # similarity = torch.mean(similarity, dim=2)
        return similarity, best_match_idx


def do_softmax(
        similarity: torch.Tensor,
        top_k: Optional[int] = None,
        inplace: bool = False,
        return_usage: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        # print("top k is not None!")
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp)  # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp)  # B*N*HW
    else:
        # similarity (B, T×HW, HW)
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        # print(f"maxes: {maxes.shape}") (B, 1, HW)
        # maxes[b, :, j] 表示：第 b 个 batch 中，query 的位置 j 在所有 memory 中最相似的那个值
        x_exp = torch.exp(similarity - maxes)
        # print(f"x_exp:{x_exp}")
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        # print(affinity)
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity


def get_affinity(mk: torch.Tensor, qk: torch.Tensor) -> torch.Tensor:
    # shorthand used in training with no top-k
    # mk.shape:torch.Size([8, 1, 384, 32, 32])
    # qk.shape:torch.Size([8, 384, 32, 32])
    mk = mk.transpose(1, 2)
    # mk.shape:torch.Size([8, 384, 1, 32, 32])
    # print(f"mk.shape:{mk.shape}")
    # print(f"qk.shape:{qk.shape}")
    similarity = get_similarity(mk, qk)
    affinity = do_softmax(similarity)
    return affinity


def readout(affinity: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
    mv = mv.transpose(1, 2).contiguous()
    # print(f"mv_shape:{mv.shape}")
    B, CV, R, H, W = mv.shape
    mo = mv.view(B, CV, R * H * W)
    # print(f"mo_shape:{mo.shape}")
    # print(f"mo_shape:{mo.shape}")
    # print(f"affinity.shape:{affinity.shape}")
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)
    return mem
