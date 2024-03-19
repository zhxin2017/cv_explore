import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm
from od import anchor
from od.config import n_cls, anchor_stride, anchor_max_size
from common.config import patch_size, grid_size_y, grid_size_x


class DetrEncoder(nn.Module):
    def __init__(self, d_enc, d_pos_emb, d_head, n_enc_layer):
        super().__init__()

        n_head = d_enc // d_head
        self.pos_emb = pe.Sinusoidal(d_pos_emb)

        self.n_enc_layer = n_enc_layer
        self.encoder_layers = nn.ModuleList()
        for i in range(n_enc_layer):
            self.encoder_layers.append(tsfm.EncoderLayer(d_enc, d_enc, n_head))

    def forward(self, x):
        b = x.shape[0]
        pos = pe.gen_pos_2d(x, pos='center').view(b, grid_size_y * grid_size_x, 2)

        pos_emb = self.pos_emb(pos)

        x = x.view(b, grid_size_y * grid_size_x, -1)
        x = torch.concat([x, pos_emb], dim=-1)

        for i in range(self.n_enc_layer):
            x = self.encoder_layers[i](x, x)
        return x


class DetrDecoder(nn.Module):
    def __init__(self, d_cls, d_obj, d_pos_emb, d_head, n_dec_layer):
        super().__init__()
        self.d_anchor = d_pos_emb * 4
        self.d_cls = d_cls
        self.d_single_cls = d_cls // n_cls
        self.d_cls_used = self.d_single_cls * n_cls
        self.d = d_cls + self.d_anchor + d_obj
        n_head = self.d // d_head

        self.n_dec_layer = n_dec_layer

        self.anchors = anchor.generate_anchors()
        self.n_anchor = len(self.anchors)
        self.anchor_emb = pe.Sinusoidal(d_pos_emb)

        self.cls_emb = pe.Embedding1D(1, d_cls)
        self.obj_emb = pe.Embedding1D(self.n_anchor, d_obj)

        self.decoder_layers = nn.ModuleList()
        for i in range(n_dec_layer):
            decoder_layer = tsfm.DecoderLayer(self.d, self.d, self.d, n_head, ommit_sa=i == 0)
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
    def __init__(self, d_cls, d_obj=128, d_pos_emb=32, d_head=128, n_enc_layer=10, n_dec_layer=8, exam_diff=True):
        super().__init__()
        d_enc = d_cls + d_obj
        d_enc_cont = d_enc - d_pos_emb * 2
        self.cnn = nn.Conv2d(3, d_enc_cont, (patch_size, patch_size), stride=(patch_size, patch_size))
        self.cnn_ln = nn.LayerNorm(d_enc_cont)

        self.encoder = DetrEncoder(d_enc, d_pos_emb, d_head, n_enc_layer)

        self.decoder = DetrDecoder(d_cls, d_obj, d_pos_emb, d_head, n_dec_layer)
        self.exam_diff = exam_diff

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)
        x = self.encoder(x)

        pos_x1y1 = pe.gen_pos_2d(x, pos='x1y1').view(x.shape[0], grid_size_y * grid_size_x, 2)
        pos_x2y2 = pe.gen_pos_2d(x, pos='x2y2').view(x.shape[0], grid_size_y * grid_size_x, 2)

        pos_x1y1_emb = self.encoder.pos_emb(pos_x1y1)
        pos_x2y2_emb = self.encoder.pos_emb(pos_x2y2)

        x = torch.concat([x, pos_x1y1_emb, pos_x2y2_emb], dim=-1)

        boxes, cls_logits = self.decoder(x)
        if self.exam_diff:
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
