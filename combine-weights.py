import torch

# from models.AnimeInterp import AnimeInterp

# model = AnimeInterp().cuda()
# print(model)

animeinterp = torch.load('checkpoints/anime_interp_full.ckpt')
d1 = animeinterp['model_state_dict']
d2 = torch.load('checkpoints/gma-sintel-no-zip.pth')

names = list(d2.keys())
new_names = []

for n in names:
    n = n[6:]
    n = "module.flownet" + n
    new_names.append(n)

for i, name in enumerate(new_names):
    if name not in d1:
        print("Adding {} from [{}]".format(name, names[i]))
        d1[name] = d2[names[i]]
    else:
        print("Replacing Weights for {} from [{}] in GMA".format(name, names[i]))
        d1[name] = d2[names[i]]

# print(animeinterp['model_state_dict'].keys())
torch.save(animeinterp, "animeinterp+gma.pth")