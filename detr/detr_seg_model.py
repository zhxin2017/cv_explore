import torch
from torch import nn
import torch.nn.functional as F
from tsfm import base, transformer
from config import ds1, ds2, downsample_seg, img_h, img_w


downsample = ds1 * ds2
fm_h = img_h // downsample
fm_w = img_w // downsample


class DetrEncoder(nn.Module):
    def __init__(self, nlayer, dmodel, dhead):
        super().__init__()
        self.cnn1 = nn.Conv2d(3, dmodel, kernel_size=ds1, stride=ds1)
        self.relu1 = nn.ReLU()
        self.cnn_ln1 = nn.LayerNorm(dmodel)
        self.cnn2 = nn.Conv2d(dmodel, dmodel, kernel_size=ds2, stride=ds2)
        self.relu2 = nn.ReLU()
        self.ln = nn.LayerNorm(dmodel)
        self.proj = nn.Linear(dmodel, dmodel)
        self.dmodel = dmodel
        self.dhead = dhead
        self.pos_y_emb_m = nn.Embedding(fm_h, dmodel)
        self.pos_x_emb_m = nn.Embedding(fm_w, dmodel)

        self.n_enc_layer = nlayer
        self.enc_layers = nn.ModuleList()
        for i in range(nlayer):
            self.enc_layers.append(transformer.TsfmLayer(dmodel, dhead))

    def forward(self, x, mask=None):
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.cnn1(x)
        x = self.relu1(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.cnn_ln1(x)
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.cnn2(x)
        x = self.relu2(x)
        x = torch.permute(x, [0, 2, 3, 1])

        n, h, w, c = x.shape
        seq_len = fm_h * fm_w
        x = x.view(n, seq_len, c)
        y_indices = torch.arange(fm_h, device=x.device)
        x_indices = torch.arange(fm_w, device=x.device)
        pos_y_emb = (self.pos_y_emb_m(y_indices).view(1, fm_h, 1, self.dmodel).
                     repeat(n, 1, fm_w, 1).view(n, seq_len, self.dmodel))
        pos_x_emb = (self.pos_x_emb_m(x_indices).view(1, 1, fm_w, self.dmodel).
                     repeat(n, fm_h, 1, 1).view(n, seq_len, self.dmodel))
        pos_emb = pos_y_emb + pos_x_emb
        x = x + pos_emb

        for enc_layer in self.enc_layers:
            x = enc_layer(x, x, x, mask)
        return x


class DetrDecoder(nn.Module):
    def __init__(self, n_dec_layer, dmodel, dhead, num_query, num_classes):
        super().__init__()
        self.n_dec_layer = n_dec_layer
        self.num_query = num_query

        self.query_emb_m = nn.Embedding(num_query, dmodel)

        self.ca_layers = nn.ModuleList()
        self.sa_layers = nn.ModuleList()
        self.dmodel = dmodel

        for i in range(n_dec_layer):
            ca_layer = transformer.TsfmLayer(dmodel, dhead)
            self.ca_layers.append(ca_layer)
            sa_layer = transformer.TsfmLayer(dmodel, dhead)
            self.sa_layers.append(sa_layer)

        self.seg_q_linear = nn.Linear(dmodel, dmodel)
        self.seg_src_linear = nn.Linear(dmodel, dmodel)
        
        self.seg_h = img_h // downsample_seg
        self.seg_w = img_w // downsample_seg
        self.num_classes = num_classes
        self.cls_linear = nn.Linear(dmodel, num_classes)

    def forward(self, src):
        n = src.shape[0]
        query_emb = self.query_emb_m(torch.arange(self.num_query, device=src.device))
        q = query_emb.view(1, self.num_query, -1).repeat(n, 1, 1)

        for i in range(self.n_dec_layer):

            q = self.ca_layers[i](q, src, src)
            q = self.sa_layers[i](q, q, q)
        q_seg = self.seg_q_linear(q)
        src_seg = self.seg_src_linear(src).permute(0, 2, 1).view(n, self.dmodel, fm_h, fm_w)
        src_seg = F.interpolate(src_seg, size=(img_h, img_w), mode='bilinear')
        src_seg = src_seg.view(n, self.dmodel, img_h * img_w)
        seg_logits = q_seg @ src_seg
        cls_logits = self.cls_linear(q)
        return seg_logits, cls_logits

class DETR(nn.Module):

    def __init__(self, dmodel, dhead, n_enc_layer, n_dec_layer, num_query, num_classes):
        super().__init__()
        self.encoder = DetrEncoder(n_enc_layer, dmodel, dhead)
        self.decoder = DetrDecoder(n_dec_layer, dmodel, dhead, num_query, num_classes)

    def forward(self, x):
        src = self.encoder(x)
        seg_logits, cls_logits = self.decoder(src)
        return seg_logits, cls_logits
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2
    imgs = torch.rand([B, 288, 512, 3])
    detr = DETR(dmodel=256, dhead=8, n_enc_layer=6, n_dec_layer=6, num_query=100, num_classes=91)
    boxes, cls_logits = detr(imgs)  
    print(boxes.shape)
    print(cls_logits.shape)
