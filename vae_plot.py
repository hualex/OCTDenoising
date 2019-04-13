import random
from skimage.measure import compare_psnr

test_dataset = []
for image,_ in dataset_train:
  test_dataset.append(image)
  
img = random.choice(test_dataset[1])
noise = torch.randn(img.shape)*noise_level

#noise  = torch.randn((1, 3, 450, 450)) * noise_level
img_n  = torch.add(img, noise)

#print(noise)
img_n = Variable(img_n).cuda()
#img_n2 = Variable(img).cuda()
denoised1,m,log = autoencoder(img_n)
#denoised2 = autoencoder(img_n2)
print(len(test_dataset))

print(compare_psnr(img.cpu().numpy(),img_n.cpu().numpy()))
print(compare_psnr(img.cpu().numpy(),denoised1.data.cpu().numpy()))
#print(compare_psnr(img.cpu().numpy(),denoised2.data.cpu().numpy()))

show_vae_img(img.numpy(), img_n.data.cpu().numpy(), denoised1.data.cpu().numpy(),(451,451))