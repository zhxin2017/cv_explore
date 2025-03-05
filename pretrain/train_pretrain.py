import os
from datetime import datetime
import torch
from torch import optim
from pretrain.pretrain_model import NextTokenPredictor 
from pretrain.pretrain_dataset import PretrainDataset
from pretrain import eval
import utils
from common.config import model_save_dir, device_type, model_save_stride, pretrain_batch_size
import time
from torch import nn
from torch.utils.data import DataLoader

device = torch.device(device_type)
model = NextTokenPredictor(dmodel=384, dhead=64, nlayer=18)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
optimizer.zero_grad()

loss_fn = nn.MSELoss(reduction='none')
train_dirs = ['/Users/zx/Documents/ml/dataset/coco/train2017', '/Users/zx/Documents/ml/dataset/VOCdevkit']
# train_dirs = ['/Users/zx/Documents/ml/dataset/coco/mini']
ds = PretrainDataset(train_dirs, random_padding=True)
dl = DataLoader(ds, batch_size=pretrain_batch_size, shuffle=True)

bp_batch_size = pretrain_batch_size
num_mini_batch = len(ds) // pretrain_batch_size
bp_num_batch = bp_batch_size // pretrain_batch_size

def train(num_epoch, log_file, output_dir):
    total_time = 0
    num_total_batch = 0
    for i in range(num_epoch):
        for j, (patches, masks, next_token_indices, loss_mask) in enumerate(dl):
            batch_start_time = time.time()
            patches = patches.to(device)
            masks = masks.to(device)
            loss_mask = loss_mask.to(device)
            next_token_indices = next_token_indices.to(device)
            patches_predict = model(patches, masks)
            b, patch_h, patch_w, c = patches.shape
            patches = patches.view(b, patch_h * patch_w, c)
            loss = loss_fn(patches_predict, patches) * loss_mask
            loss = loss.mean() * patch_h * patch_w
            t = time.time()
            loss.backward()
            t_bp = time.time() - t

            if (j + 1) % bp_num_batch == 0:
                optimizer.step()
                optimizer.zero_grad()
            # nn.utils.clip_grad_value_(tsfm.parameters(), 0.05)
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            total_time += batch_duration
            num_total_batch += 1
            avg_batch_time = total_time / num_total_batch
            num_batch_left = num_mini_batch - (j + 1)
            epoch_time_left = num_batch_left * avg_batch_time
            if (j + 1) % model_save_stride == 0:
                torch.save(model.state_dict(), f'{model_save_dir}/pretrain_e{i + 1}_i{j + 1}.pt')
                eval.eval(model, output_dir, i + 1, j + 1)
            
            rec_info = (f'|epoch {i + 1}/{num_epoch}|batch {j}/{num_mini_batch}|'
                f'loss {loss.detach().item():.3f}|tbp {t_bp:.3f}|'
                f'time {batch_duration:.3f}, avg {avg_batch_time:.3f}, '
                f'left {utils.seconds_to_str(epoch_time_left)}')
            print(rec_info)
            with open(log_file, 'a') as f:
                print(rec_info, file=f)


if __name__ == '__main__':
    # model_files = os.listdir(model_save_dir)
    # model_files = [f for f in model_files if f.endswith('.pt') and f.startswith('pretrain')]
    # if len(model_files) == 0:
    #     model_path_old = None
    #     latest_version = 0
    # else:
    #     versions = [int(f.split('.')[0].split('_')[-1]) for f in model_files]
    #     latest_version = max(versions)
    #     model_path_old = f'{model_save_dir}/pretrain_{latest_version}.pt'
    #     saved_state = torch.load(model_path_old, map_location=device)
    #     model.load_state_dict(saved_state)
    #     # state = tsfm.state_dict()
    #     # for k in state.keys():
    #     #     if k in saved_state:
    #     #         state[k] = saved_state[k]
    #     # tsfm.load_state_dict(state)
    model_file = 'outputs/pretrain/pretrain_e4_i3600.pt'
    saved_state = torch.load(model_file, map_location=device)
    model.load_state_dict(saved_state)
    num_epoch = 200
    print(os.getcwd())
    now = datetime.strftime(datetime.now(), '%m%d%H%M%S')
    output_dir = f'outputs/pretrain/{now}'
    os.mkdir(output_dir)
    log_file = f'{output_dir}/pretrain_log.txt'
    train(num_epoch, log_file, output_dir)
    # model_path_new = f'{model_save_dir}/pretrain_{n_smp}.pt'
    # torch.save(model.state_dict(), model_path_new)
    # if model_path_old is not None:
    #     os.remove(model_path_old)
    # model_path_old = model_path_new
        # break
