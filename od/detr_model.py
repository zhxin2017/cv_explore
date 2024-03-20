import random

import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm
from od import anchor, box
from od.config import n_cls, anchor_stride, anchor_max_size
from common.config import patch_size, grid_size_y, grid_size_x


class DetrEncoder(nn.Module):
    def __init__(self, d_enc, d_pos_emb, n_head, d_head, n_enc_layer):
        super().__init__()

        self.pos_emb = pe.Sinusoidal(d_pos_emb)

        self.n_enc_layer = n_enc_layer
        self.encoder_layers = nn.ModuleList()
        for i in range(n_enc_layer):
            self.encoder_layers.append(tsfm.EncoderLayer(d_enc, d_enc, n_head, d_head))

    def forward(self, x):
        b = x.shape[0]
        pos = pe.gen_pos_2d(x, pos='center').view(b, grid_size_y * grid_size_x, 2)

        pos_emb = self.pos_emb(pos)

        x = x.view(b, grid_size_y * grid_size_x, -1)
        x = torch.concat([x, pos_emb], dim=-1)

        for i in range(self.n_enc_layer):
            x = self.encoder_layers[i](x, x)
        return x


class DetrDecoder2(nn.Module):
    def __init__(self, d_pos_emb, d_cls, d_obj, d_extremity, train=False):
        super().__init__()
        self.d_anchor = d_pos_emb * 4
        self.d_cont = d_cls + d_obj
        self.d_cls = d_cls
        self.d_extremity = d_extremity
        self.d_delta = d_pos_emb * 2
        self.d_v = d_cls + d_obj + d_extremity + self.d_delta
        self.pos_emb = pe.Sinusoidal(d_pos_emb)
        self.extremity_x1_emb = pe.Embedding1D(1, d_extremity)
        self.extremity_y1_emb = pe.Embedding1D(1, d_extremity)
        self.extremity_x2_emb = pe.Embedding1D(1, d_extremity)
        self.extremity_y2_emb = pe.Embedding1D(1, d_extremity)

        self.anchors = anchor.generate_anchors()
        self.n_anchor = len(self.anchors)
        self.anchor_emb = pe.Sinusoidal(d_pos_emb)
        self.cont_finder = tsfm.MHA(self.d_anchor, self.d_anchor, self.d_cont,
                                    n_head=8, d_head=self.d_anchor, project_v=False)

        d_extremity_finder = d_pos_emb + d_cls + d_obj + d_extremity
        self.extremity_finder = tsfm.MHA(d_extremity_finder, d_extremity_finder, self.d_v,
                                          n_head=1, d_head=d_extremity_finder, project_v=False)

        self.train = train
        self.cls_reg = nn.Linear(d_cls, n_cls)
        d_delta_reg = d_extremity + self.d_delta
        self.box_delta_reg = base.MLP(d_delta_reg, d_delta_reg * 2, 1, 2)

    def extremity_decoder(self, anchor_extremity, memory_extremity, extremity_emb, memory, contents):
        anchor_x1_emb = self.pos_emb(anchor_extremity)
        memory_pos_x1_emb = self.pos_emb(memory_extremity)
        extremity_x1_emb = extremity_emb(memory).repeat(1, self.n_anchor, 1)
        q_extremity = torch.concat([contents, extremity_x1_emb, anchor_x1_emb], dim=-1)
        k_extremity = torch.concat([memory[..., :self.d_cont + self.d_extremity], memory_pos_x1_emb], dim=-1)
        extremity_content = self.extremity_finder(q_extremity, k_extremity, memory)
        box_delta = F.tanh(self.box_delta_reg(extremity_content[..., self.d_cont:])) / (anchor_max_size / anchor_stride) / 2
        return extremity_content, box_delta

    def forward(self, memory):
        B = memory.shape[0]
        anchors = torch.tensor(self.anchors, device=memory.device).unsqueeze(0).repeat(B, 1, 1)

        anchor_emb = self.anchor_emb(anchors)
        memory_pos_x1y1 = pe.gen_pos_2d(memory, pos='x1y1').view(B, grid_size_y * grid_size_x, 2)
        memory_pos_x2y2 = pe.gen_pos_2d(memory, pos='x2y2').view(B, grid_size_y * grid_size_x, 2)
        memory_pos_emb = self.pos_emb(torch.concat([memory_pos_x1y1, memory_pos_x2y2], dim=-1))

        contents = self.cont_finder(anchor_emb, memory_pos_emb, memory[..., :self.d_cont])

        extremity_content_x1, box_delta_x1 = self.extremity_decoder(anchors[:, :, 0:1], memory_pos_x1y1[:, :, 0:1], self.extremity_x1_emb, memory, contents)
        extremity_content_y1, box_delta_y1 = self.extremity_decoder(anchors[:, :, 1:2], memory_pos_x1y1[:, :, 1:2], self.extremity_y1_emb, memory, contents)
        extremity_content_x2, box_delta_x2 = self.extremity_decoder(anchors[:, :, 2:3], memory_pos_x2y2[:, :, 0:1], self.extremity_x2_emb, memory, contents)
        extremity_content_y2, box_delta_y2 = self.extremity_decoder(anchors[:, :, 3:4], memory_pos_x2y2[:, :, 1:2], self.extremity_y2_emb, memory, contents)
        # extremity_contents = torch.concat([extremity_content_x1, extremity_content_y1, extremity_content_x2, extremity_content_y2], dim=-1).view(B, self.n_anchor, 4, self.d_v)
        box_delta = torch.concat([box_delta_x1, box_delta_y1, box_delta_x2, box_delta_y2], dim=-1)
        # aggre_indices = [0, 1, 2, 3]
        #
        # if self.train:
        #     n_sample = random.choice([1, 2, 3, 4])
        #     aggre_indices = random.sample(aggre_indices, n_sample)
        #
        # aggre_content = extremity_contents[:, :, aggre_indices].mean(dim=-2)
        # cls_logits = self.cls_reg(aggre_content[..., :self.d_cls])
        cls_logits = self.cls_reg(contents[..., :self.d_cls])

        boxes = box_delta + anchors

        return boxes, cls_logits


class DetrDecoder(nn.Module):
    def __init__(self, d_cls, d_obj, d_pos_emb, n_head, d_head, n_dec_layer):
        super().__init__()
        self.d_anchor = d_pos_emb * 4
        self.d_cls = d_cls
        self.d_single_cls = d_cls // n_cls
        self.d_cls_used = self.d_single_cls * n_cls
        self.d = d_cls + self.d_anchor + d_obj

        self.n_dec_layer = n_dec_layer

        self.anchors = anchor.generate_anchors()
        self.n_anchor = len(self.anchors)
        self.anchor_emb = pe.Sinusoidal(d_pos_emb)

        self.cls_emb = pe.Embedding1D(1, d_cls)
        self.obj_emb = pe.Embedding1D(self.n_anchor, d_obj)

        self.decoder_layers = nn.ModuleList()
        for i in range(n_dec_layer):
            decoder_layer = tsfm.DecoderLayer(self.d, self.d, self.d, n_head, d_head, ommit_sa=i == 0)
            self.decoder_layers.append(decoder_layer)

        self.box_delta_reg = base.MLP(self.d, self.d * 2, 4, 2)

    def forward(self, memory):
        B = memory.shape[0]
        boxes = torch.tensor(self.anchors, device=memory.device).unsqueeze(0).repeat(B, 1, 1)
        ''' 
        # don't have enough computing power, the query number is huge.
        
        anchors_emb = self.anchor_emb(boxes).view(B, self.n_anchor, 1, self.d_anchor).\
            expand(B, self.n_anchor, category_num, self.d_anchor)
        cls_emb = self.cls_emb(memory).view(B, 1, category_num, self.d_cont).\
            expand(B, self.n_anchor, category_num, self.d_cont)

        q = torch.concat([cls_emb, anchors_emb], dim=-1).view(B, self.n_anchor * category_num, self.d)
        print(q.shape)
        '''

        cls_emb = self.cls_emb(memory)
        q_cls = cls_emb.expand(B, self.n_anchor, self.d_cls)

        cls_reg_mask = F.one_hot(torch.arange(0, n_cls, device=memory.device), num_classes=n_cls)
        cls_reg_mask = cls_reg_mask.view(1, n_cls, 1, n_cls).expand(1, n_cls, self.d_single_cls, n_cls). \
            reshape(1, 1, self.d_cls_used, n_cls)

        cls_reg_weight = cls_reg_mask * (q_cls.view(B, self.n_anchor, self.d_cls, 1)[:, :, :self.d_cls_used])

        q_obj = self.obj_emb(memory)
        q_anchor = self.anchor_emb(boxes).view(B, self.n_anchor, self.d_anchor)
        q = torch.concat([q_cls, q_obj, q_anchor], dim=-1).view(B, self.n_anchor, self.d)

        for i in range(self.n_dec_layer):
            q = self.decoder_layers[i](q, memory, memory)

        box_delta = F.tanh(self.box_delta_reg(q)) / (anchor_max_size / anchor_stride) / 2
        boxes = box_delta + boxes

        # (B, n_anchor, 1, n_cls_used)  @ (B, n_anchor, d_cls_used, n_cls)
        cls_logits = q[:, :, :self.d_cls_used].view(B, self.n_anchor, 1, self.d_cls_used) @ cls_reg_weight
        cls_logits = cls_logits.view(B, self.n_anchor, n_cls)

        return boxes, cls_logits


class DETR(nn.Module):

    def __init__(self, d_cls=256, d_obj=64, d_extremity=64, d_pos_emb=32, d_enc_head=128,
                 n_enc_head=8, n_enc_layer=16, exam_diff=True, train=True):
        super().__init__()
        d_enc_cont = d_cls + d_obj + d_extremity
        d_enc = d_enc_cont + d_pos_emb * 2
        self.cnn = nn.Conv2d(3, d_enc_cont, (patch_size, patch_size), stride=(patch_size, patch_size))
        self.cnn_ln = nn.LayerNorm(d_enc_cont)

        self.encoder = DetrEncoder(d_enc, d_pos_emb, n_enc_head, d_enc_head, n_enc_layer)

        self.decoder = DetrDecoder2(d_pos_emb, d_cls, d_obj, d_extremity, train)
        self.exam_diff = exam_diff

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)
        x = self.encoder(x)

        boxes, cls_logits = self.decoder(x)

        if self.exam_diff and x.shape[0] > 1:
            enc_diff = (x[0] - x[1]).abs().mean()
            logits_diff = (cls_logits[0] - cls_logits[1]).abs().mean()
        else:
            enc_diff = 0
            logits_diff = 0

        return boxes, cls_logits, enc_diff, logits_diff


if __name__ == '__main__':
    device = torch.device('mps')

    imgs = torch.rand([2, 3, 512, 512], device=device)
    detr = DETR(256, device=device)
    detr.to(device=device)
    categories, anchors = detr(imgs)
