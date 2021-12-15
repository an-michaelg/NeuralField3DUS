"""
Meta-learning and training for fitting 3D volumes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
#from pathlib import Path

from dataloader import get_dataset_paths, get_2d_dataloader
from dataloader import get_3d_dataloader
from siren import Siren

plt.rcParams['image.cmap'] = 'gray'
torch.cuda.empty_cache()
torch.manual_seed(42)

#%%
config_dict = {
    "outer_loop_steps": 100001,
    "outer_steps_til_summary": 10000,
    "outer_batch_size": 4,
    "outer_momentum": (0.9, 0.999),
    "optimizer": "Adam",
    "lookahead_steps": 10,
    "omega": 240,
    "dim_option": "3D_slice",
    "meta_lr_0": 1e-5,
    "adam_lr": 1e-4,
    "total_steps": 20001,
    "steps_til_summary": 1000
    }
import wandb
run = wandb.init(project="meta-3dus", entity="guangnan", config=config_dict)
#%% instantiate the siren model, input = (x,y), output = intensity
sl = 208 # sidelength of image input
epsilon = 1e-8

img_siren = Siren(in_features=3, out_features=1, hidden_features=sl, 
                  hidden_layers=3, outermost_linear=True, 
                  first_omega_0=wandb.config["omega"])
img_siren.cuda()
if wandb.config["optimizer"] == "Adam":
    optim = torch.optim.Adam(lr=wandb.config["adam_lr"], 
                              betas=wandb.config["outer_momentum"],
                              params=img_siren.parameters())
elif wandb.config["optimizer"] == "SGD":
    optim = torch.optim.SGD(lr=wandb.config["adam_lr"],
                            params=img_siren.parameters())
else:
    raise NotImplementedError()

#%% train the meta-learning portion of siren 
# ("inspired" by https://openai.com/blog/reptile/)

list_img_path_c = get_dataset_paths(dataset="US", data_option="C")
outer_loader = get_2d_dataloader(sl, wandb.config["outer_batch_size"], True, 
                                 list_img_path_c, wandb.config["dim_option"])

for outstep in range(wandb.config["outer_loop_steps"]):
    initial_weights = deepcopy(img_siren.state_dict())
    model_input, ground_truth = next(iter(outer_loader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    # train inner loop steps
    for instep in range(wandb.config["lookahead_steps"]):
        model_output, coords = img_siren(model_input)
        loss = ((model_output - ground_truth)**2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    # graft the trained weights onto initial weights (reptile step)
    meta_lr = wandb.config["meta_lr_0"] #* (1 - outstep/outer_loop_steps)
    final_weights = img_siren.state_dict()
    img_siren.load_state_dict({name : 
        initial_weights[name] + meta_lr*(final_weights[name] - initial_weights[name]) 
        for name in initial_weights})
    # visualize the meta-learned output
    if not outstep % wandb.config["outer_steps_til_summary"]:
        print("Outer step %d" % (outstep))
        plt.imshow(model_output[0].cpu().view(sl,sl).detach().numpy())
        plt.show()

if wandb.config["outer_loop_steps"] > 10000:
    model_save_dir = os.path.join('../runs_3d/' + wandb.run.name)
    os.makedirs(model_save_dir)
    torch.save(img_siren.state_dict(), os.path.join(model_save_dir, 'state_dict_meta.pt'))

#%% train the image-fitting portion of siren

#img_path_p = get_dataset_paths(dataset="US", data_option="P")
#dataloader = get_dataloader(sl, 1, False, img_path_p)
dataloader = get_3d_dataloader(4, True, '../3dus_example.npy')
valid_loader = get_3d_dataloader(4, True, '../3dus_example.npy')
# if wandb.config["optimizer"] == "Adam":
#     # re-enable the momentum for optimization on a single object
#     optim.param_groups[0]['betas'] = (0.9, 0.999)
# Since the whole image is our dataset, this just means 500 gradient descent steps.
for step in range(wandb.config["total_steps"]):
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    model_output, coords = img_siren(model_input)
    
    loss = ((model_output - ground_truth)**2).mean()
    # PSNR = 10*log10(MAX/loss)
    img_max = (torch.max(ground_truth) - torch.min(ground_truth)).cpu()
    psnr = 20 * np.log10(img_max) - 10 * np.log10(loss.cpu().detach())
    
    wandb.log({"loss": loss})
    wandb.log({"psnr": psnr})
    
    if not step % wandb.config["steps_til_summary"]:
        # acquire random slices of 3D volume
        valid_in, valid_gt = next(iter(valid_loader))
        valid_in, valid_gt = valid_in.cuda(), valid_gt.cuda()
        valid_out, coords2 = img_siren(valid_in)
        
        val_loss = ((valid_out - valid_gt)**2).mean()
        val_img_max = (torch.max(valid_gt) - torch.min(valid_gt)).cpu()
        val_psnr = 20 * np.log10(val_img_max) - 10 * np.log10(val_loss.cpu().detach())  
        
        print("Step %d, Train: loss %0.6f, PSNR %.3f, Val: loss %0.6f, PSNR %.3f" 
              % (step, loss, psnr, val_loss, val_psnr))
        

        fig, axes = plt.subplots(1,2, figsize=(13,6))
        axes[0].imshow(ground_truth[0].cpu().view(sl,sl).numpy())
        axes[1].imshow(model_output[0].cpu().view(sl,sl).detach().numpy())
        #axes[1].imshow(img_grad.norm(dim=-1).cpu().view(sl,sl).detach().numpy())
        #axes[2].imshow(img_laplacian.cpu().view(sl,sl).detach().numpy())
        plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()

if wandb.config["total_steps"] > 10000:
    model_save_dir = os.path.join('../runs_3d/' + wandb.run.name)
    if wandb.config["outer_loop_steps"] <= 10000:
        os.makedirs(model_save_dir)
    torch.save(img_siren.state_dict(), os.path.join(model_save_dir, 'state_dict_trained.pt'))
    
run.finish()

#%% test the model in a uniform interval over the slices
valid_loader = get_3d_dataloader(1, False, '../3dus_example.npy')
psnr_arr = []
frames = []
max_frame = 150
for i, data in enumerate(valid_loader):
    model_in, gt = data
    model_in, gt = model_in.cuda(), gt.cuda()
    model_out, coords = img_siren(model_in)
    loss = ((model_out - gt)**2).mean()
    img_max = (torch.max(gt) - torch.min(gt)).cpu() + epsilon
    psnr = 20 * np.log10(img_max) - 10 * np.log10(loss.cpu().detach())
    psnr_arr.append(psnr)
    
    fig, axes = plt.subplots(1,2, figsize=(18,6))
    axes[0].imshow(gt.cpu().view(sl,sl).numpy())
    axes[1].imshow(model_out.cpu().view(sl,sl).detach().numpy())
    title = "File: 3dus_example.py, Frame = {0}/{1}, Frame PSNR: {2:.3f}, " \
        .format(i, max_frame, psnr)
    fig.suptitle(title)
    plt.savefig("{0}.jpg".format(i))
    print(title)
    
    if i >= max_frame:
        break