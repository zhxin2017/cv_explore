import torch
from torch import nn
import torch.nn.functional as F
from tsfm import base, transformer
from config import downsample1, downsample2, img_h, img_w


downsample = downsample1 * downsample2
fm_h = img_h // downsample
fm_w = img_w // downsample


class DetrEncoder(nn.Module):
    def __init__(self, nlayer, dmodel, dhead):
        super().__init__()
        self.cnn1 = nn.Conv2d(3, dmodel, kernel_size=downsample1, stride=downsample1)
        self.relu1 = nn.ReLU()
        self.cnn_ln1 = nn.LayerNorm(dmodel)
        self.cnn2 = nn.Conv2d(dmodel, dmodel, kernel_size=downsample2, stride=downsample2)
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
        seq_len = h * w
        x = x.view(n, seq_len, c)
        y_indices = torch.arange(h, device=x.device)
        x_indices = torch.arange(w, device=x.device)
        pos_y_emb = (self.pos_y_emb_m(y_indices).view(1, h, 1, self.dmodel).
                     repeat(n, 1, w, 1).view(n, seq_len, self.dmodel))
        pos_x_emb = (self.pos_x_emb_m(x_indices).view(1, 1, w, self.dmodel).
                     repeat(n, h, 1, 1).view(n, seq_len, self.dmodel))
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

        for i in range(n_dec_layer):
            ca_layer = transformer.TsfmLayer(dmodel, dhead)
            self.ca_layers.append(ca_layer)
            sa_layer = transformer.TsfmLayer(dmodel, dhead)
            self.sa_layers.append(sa_layer)

        self.box_linear1 = nn.Linear(dmodel, dmodel // 2)
        self.box_relu = nn.ReLU()
        self.box_linear2 = nn.Linear(dmodel // 2, 4)
        self.box_sigmoid = nn.Sigmoid()
        self.cls_linear = nn.Linear(dmodel, num_classes)

    def forward(self, src):
        n = src.shape[0]
        query_emb = self.query_emb_m(torch.arange(self.num_query, device=src.device))
        q = query_emb.view(1, self.num_query, -1).repeat(n, 1, 1)

        for i in range(self.n_dec_layer):

            q = self.ca_layers[i](q, src, src)
            q = self.sa_layers[i](q, q, q)

        boxes = self.box_linear1(q)
        boxes = self.box_relu(boxes)
        boxes = self.box_linear2(boxes)
        boxes = self.box_sigmoid(boxes)
        cls_logits = self.cls_linear(q)
        return boxes, cls_logits


class DETR(nn.Module):

    def __init__(self, dmodel, dhead, n_enc_layer, n_dec_layer, num_query, num_classes):
        super().__init__()
        self.encoder = DetrEncoder(n_enc_layer, dmodel, dhead)
        self.decoder = DetrDecoder(n_dec_layer, dmodel, dhead, num_query, num_classes)

    def forward(self, x):
        src = self.encoder(x)
        boxes, cls_logits = self.decoder(src)
        return boxes, cls_logits

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2
    imgs = torch.rand([B, 288, 512, 3])
    detr = DETR(dmodel=256, dhead=8, n_enc_layer=6, n_dec_layer=6, num_query=100, num_classes=91)
    boxes, cls_logits = detr(imgs)  
    print(boxes.shape)
    print(cls_logits.shape)
