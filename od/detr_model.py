import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm
from od import anchor, config


class DetrEncoder(nn.Module):
    def __init__(self, n_head, d_cont, d_pos, n_enc_layer):
        super().__init__()
        d_q = d_cont + d_pos
        self.d_pos = d_pos
        self.encoder_layers = nn.ModuleList()
        for i in range(n_enc_layer):
            decoder_layer = tsfm.EncoderLayer(d_q, d_cont, n_head)
            self.encoder_layers.append(decoder_layer)
        self.pos_emb = None

    def forward(self, x):
        B, H, W, C = x.shape
        coord = pe.gen_pos_2d(x).view(B, H * W, 2)
        pos_emb = pe.sinusoidal_encoding(coord, self.d_pos // 2)
        self.pos_emb = pos_emb
        x = x.view(B, H * W, -1)

        for enc in self.encoder_layers:
            qk = torch.concat([x, pos_emb], dim=-1)
            x = enc(qk, x)

        return x


class DetrDecoder(nn.Module):
    def __init__(self, n_head, d_v, d_anchor, n_dec_layer):
        super().__init__()
        d_q = d_v + d_anchor
        self.d_anchor = d_anchor
        self.d_v = d_v
        self.decoder_layers = nn.ModuleList()
        for i in range(n_dec_layer):
            decoder_layer = tsfm.DecoderLayer(d_q, d_v, d_v, n_head, ommit_sa=i==0)
            self.decoder_layers.append(decoder_layer)
        self.q_cont_emb = pe.Embedding1D(1, self.d_v)
        self.coord_delta_reg = base.MLP(d_v, d_v * 2, 4, 2)
        self.q_ln = nn.LayerNorm(d_v)
        self.cls_reg = base.MLP(d_v, d_v * 2, config.category_num, 2)

    def forward(self, anchors, memory):
        B = memory.shape[0]
        boxes = anchors.unsqueeze(0).repeat(B, 1, 1)
        anchors_emb = pe.sinusoidal_encoding(boxes, self.d_anchor // 4)
        q_cont = torch.zeros(B, config.n_query, self.d_v, device=memory.device)
        # q_cont = self.q_cont_emb(memory).repeat(1, config.n_query, 1)
        q = torch.concat([q_cont, anchors_emb], dim=-1)
        for dec in self.decoder_layers:
            q_cont = dec(q, memory, memory)
            anchors_delta = F.tanh(self.coord_delta_reg(self.q_ln(q_cont))) / 16
            boxes = anchors_delta + boxes
            anchors_emb = pe.sinusoidal_encoding(boxes, self.d_anchor // 4)
            q = torch.concat([q_cont, anchors_emb], dim=-1)

        cls_logits = self.cls_reg(self.q_ln(q_cont))

        return boxes, cls_logits


class DETR(nn.Module):
    def __init__(self, d_cont, d_pos, d_anchor, n_head=8, n_enc_layer=10, n_dec_layer=8, device=torch.device('mps')):
        super().__init__()
        self.d_cont = d_cont
        self.d_coord = d_cont // 2
        self.n_dec_layer = n_dec_layer
        self.device = device
        self.cnn1 = nn.Conv2d(3, d_cont, (2, 2), stride=(2, 2))
        self.cnn2 = nn.Conv2d(d_cont, d_cont, (2, 2), stride=(2, 2))
        self.cnn3 = nn.Conv2d(d_cont, d_cont, (2, 2), stride=(2, 2))
        self.cnn4 = nn.Conv2d(d_cont, d_cont, (2, 2), stride=(2, 2))
        self.cnn_ln = nn.LayerNorm(d_cont)

        self.encoder = DetrEncoder(n_head, d_cont, d_pos, n_enc_layer)
        anchors = anchor.generate_anchors()
        anchors = torch.tensor(anchors, device=device)
        self.anchors = anchors

        self.decoder = DetrDecoder(n_head, d_cont + d_pos, d_anchor, n_dec_layer)

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = F.relu(self.cnn4(x))
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)

        x = self.encoder(x)
        x = torch.concat([x, self.encoder.pos_emb], dim=-1)
        boxes, cls_logits = self.decoder(self.anchors, x)

        return boxes, cls_logits


if __name__ == '__main__':
    device = torch.device('mps')

    imgs = torch.rand([2, 3, 512, 512], device=device)
    detr = DETR(256, device=device)
    detr.to(device=device)
    categories, anchors = detr(imgs)
