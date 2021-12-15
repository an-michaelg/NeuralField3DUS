'''
Training for meta-learning on SIREN models
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
#from pathlib import Path

from dataloader import get_dataset_paths, get_dataloader
from siren import Siren, gradient, laplace

plt.rcParams['image.cmap'] = 'gray'
torch.cuda.empty_cache()
torch.manual_seed(42)

#%%
config_dict = {
    "outer_loop_steps": 0,
    "outer_steps_til_summary": 1000,
    "outer_batch_size": 3,
    "outer_momentum": (0.9, 0.999),
    "optimizer": "Adam",
    "lookahead_steps": 4,
    "omega": 120,
    "meta_lr_0": 1e-5,
    "adam_lr": 5e-4,
    "total_steps": 61,
    "steps_til_summary": 10
    }
import wandb
run=wandb.init(project="meta-us", entity="guangnan", config=config_dict)
#%% instantiate the siren model, input = (x,y), output = intensity

img_siren = Siren(in_features=2, out_features=1, hidden_features=256, 
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
outer_loader = get_dataloader(256, wandb.config["outer_batch_size"], True, list_img_path_c)

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
        plt.imshow(model_output[0].cpu().view(256,256).detach().numpy())
        plt.show()

if wandb.config["outer_loop_steps"] > 10000:
    model_save_dir = os.path.join('../runs/' + wandb.run.name)
    os.makedirs(model_save_dir)
    torch.save(img_siren.state_dict(), os.path.join(model_save_dir, 'state_dict.pt'))
    
#%% load the model
load_model = False
if load_model:
    model_load_dir = '../runs/'
    load_model_path = os.path.join(model_load_dir, 'meta_5e4/state_dict.pt')
    img_siren.load_state_dict(torch.load(load_model_path))

#%% train the image-fitting portion of siren

img_path_p = get_dataset_paths(dataset="US", data_option="C")
dataloader = get_dataloader(256, 1, False, img_path_p)
model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

# if wandb.config["optimizer"] == "Adam":
#     # re-enable the momentum for optimization on a single object
#     optim.param_groups[0]['betas'] = (0.9, 0.999)
# Since the whole image is our dataset, this just means 500 gradient descent steps.
for step in range(wandb.config["total_steps"]):
    model_output, coords = img_siren(model_input)    
    loss = ((model_output - ground_truth)**2).mean()
    # PSNR = 10*log10(MAX/loss)
    img_max = (torch.max(ground_truth) - torch.min(ground_truth)).cpu()
    psnr = 20 * np.log10(img_max) - 10 * np.log10(loss.cpu().detach())
    
    wandb.log({"loss": loss})
    wandb.log({"psnr": psnr})
    
    if not step % wandb.config["steps_til_summary"]:
        print("Step %d, Total loss %0.6f, PSNR %.3f" % (step, loss, psnr))
        img_grad = gradient(model_output, coords)
        img_laplacian = laplace(model_output, coords)

        fig, axes = plt.subplots(1,3, figsize=(18,6))
        axes[0].imshow(model_output.cpu().view(256,256).detach().numpy())
        axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256,256).detach().numpy())
        axes[2].imshow(img_laplacian.cpu().view(256,256).detach().numpy())
        plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()
run.finish()