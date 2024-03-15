import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm
from od import anchor, config
from common.config import patch_size


class DetrEncoder(nn.Module):
    def __init__(self, n_head, d_cont, d_pos, n_enc_layer):
        super().__init__()
        d_q = d_cont + d_pos
        self.d_pos = d_pos

        self.pos_emb = pe.Sinusoidal(self.d_pos // 2)

        self.n_enc_layer = n_enc_layer
        self.encoder_layers = nn.ModuleList()
        for i in range(n_enc_layer):
            self.encoder_layers.append(tsfm.EncoderLayer(d_q, d_cont, n_head))

        self.cont_ln = nn.LayerNorm(d_cont)
        self.pos_delta_reg = base.MLP(d_cont, d_cont * 2, 2, 2)

    def forward(self, x):
        B, H, W, C = x.shape
        coord = pe.gen_pos_2d(x).view(B, H * W, 2)
        pos_emb = self.pos_emb(coord)
        x = x.view(B, H * W, -1)

        for i in range(self.n_enc_layer):
            qk = torch.concat([x, pos_emb], dim=-1)
            x = self.encoder_layers[i](qk, x)

        coord = coord + F.tanh(self.pos_delta_reg(self.cont_ln(x))) / max(H, W) / 2
        pos_emb = self.pos_emb(coord)

        return torch.concat([x, pos_emb], dim=-1)


class DetrDecoder(nn.Module):
    def __init__(self, n_head, d_v, d_anchor, n_dec_layer):
        super().__init__()
        d_q = d_v + d_anchor
        self.n_dec_layer = n_dec_layer
        self.d_anchor = d_anchor

        self.anchor_emb = pe.Sinusoidal(self.d_anchor // 4)

        self.d_v = d_v
        self.decoder_layers = nn.ModuleList()
        self.coord_delta_regs = nn.ModuleList()
        for i in range(n_dec_layer):
            decoder_layer = tsfm.DecoderLayer(d_q, d_v, d_v, n_head, ommit_sa=i==0)
            coord_delta_reg = base.MLP(d_v, d_v * 2, 4, 2)
            self.decoder_layers.append(decoder_layer)
            self.coord_delta_regs.append(coord_delta_reg)
        # self.q_cont_emb = pe.Embedding1D(1, self.d_v)
        self.q_ln = nn.LayerNorm(d_v)
        self.cls_reg = base.MLP(d_v, d_v * 2, config.category_num, 2)
        self.anchors = anchor.generate_anchors()

    def forward(self, memory):
        B = memory.shape[0]
        boxes = torch.tensor(self.anchors, device=memory.device).unsqueeze(0).repeat(B, 1, 1)
        anchors_emb = self.anchor_emb(boxes)
        q_cont = torch.zeros(B, config.n_query, self.d_v, device=memory.device)
        # q_cont = self.q_cont_emb(memory).repeat(1, config.n_query, 1)
        q = torch.concat([q_cont, anchors_emb], dim=-1)
        for i in range(self.n_dec_layer):
            q_cont = self.decoder_layers[i](q, memory, memory)
            anchors_delta = F.tanh(self.coord_delta_regs[i](self.q_ln(q_cont))) / 16
            boxes = anchors_delta + boxes
            anchors_emb = self.anchor_emb(boxes)
            q = torch.concat([q_cont, anchors_emb], dim=-1)

        cls_logits = self.cls_reg(self.q_ln(q_cont))

        return boxes, cls_logits


class DETR(nn.Module):
    def __init__(self, d_cont, d_pos, d_anchor, n_head=8, n_enc_layer=10, n_dec_layer=8):
        super().__init__()
        self.d_cont = d_cont
        self.d_coord = d_cont // 2
        self.n_dec_layer = n_dec_layer
        self.cnn = nn.Conv2d(3, d_cont, (patch_size, patch_size), stride=(patch_size, patch_size))
        self.cnn_ln = nn.LayerNorm(d_cont)

        self.encoder = DetrEncoder(n_head, d_cont, d_pos, n_enc_layer)

        self.decoder = DetrDecoder(n_head, d_cont + d_pos, d_anchor, n_dec_layer)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)

        x = self.encoder(x)
        boxes, cls_logits = self.decoder(x)

        return boxes, cls_logits


if __name__ == '__main__':
    device = torch.device('mps')

    imgs = torch.rand([2, 3, 512, 512], device=device)
    detr = DETR(256, device=device)
    detr.to(device=device)
    categories, anchors = detr(imgs)
