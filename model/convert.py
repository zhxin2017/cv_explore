import sys
sys.path.append('..')
from model import enc
from common.config import device_type, model_save_dir
import torch
from ssl_ import ssl_model

device = torch.device(device_type)
ssl_model_ = ssl_model.SSL(d_cont=384, d_coord_emb=64, d_head=64, n_enc_layer=20)
saved_state = torch.load('/Users/zx/Documents/ml/restart/resources/ssl_410.pt', map_location=device)
ssl_model_.load_state_dict(saved_state)


encoder = enc.Encoder(n_enc_layer=20, d_cont=384, d_head=64, d_coord_emb=64)

enc_state = encoder.state_dict()
ssl_state = ssl_model_.encoder.state_dict()
for k in enc_state.keys():
    if k in ssl_state:
        enc_state[k] = ssl_state[k]

torch.save(enc_state, f'{model_save_dir}/ssl_enc_410.pt')