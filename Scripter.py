import torch
import matplotlib.pyplot as plt
inmage = torch.load('in_imageone.pt')

outmage = torch.load('out_imageone.pt')



print(torch.mean(inmage))
print(torch.std(inmage))
print(outmage)6
print(torch.kl_div(outmage, inmage))
plt.plot