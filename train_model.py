

train_loss = []
valid_loss = []
time_epoch = []
gpu_size = []
v_gpu_size = []

epoche_number = 10
for i in range(epoche_number):
    
    # Let's train the model
    s = time.time() 
    total_loss = 0.0
    total_iter = 0
    j = 0
    autoencoder.train()
    for image, label in dataset_train_loader:      
        j+=1      
        #print('Batch iter: {} Beging traning GPU Memory allocated: {} MB'.format(j,torch.cuda.memory_allocated() / 1024**2))
        gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        noise = torch.randn(image.shape[0], 3, 450, 450) * noise_level
        image_n = torch.add(image, noise)
        image = Variable(image).cuda()
        image_n = Variable(image_n).cuda()
        #print('Batch iter: {} before training traning GPU Memory allocated: {} MB'.format(j,torch.cuda.memory_allocated() / 1024**2))
        gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
       

        optimizer.zero_grad()
        output = autoencoder(image_n)
        
        loss = loss_func(output, image)
        loss.backward()
        #print('Batch iter: {} after training traning GPU Memory allocated: {} MB'.format(j,torch.cuda.memory_allocated() / 1024**2))
        gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        optimizer.step()
        del image
        del image_n
        
        total_iter += 1
        total_loss += loss.item()
    #print('Epoch:{} GPU Memory allocated: {} MB'.format(i,torch.cuda.memory_allocated() / 1024**2))
        
     
        
    # Let's record the validation loss
    
    total_val_loss = 0.0
    total_val_iter = 0
    autoencoder.eval()
    for image, label in dataset_valid_loader:
        v_gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
      
      
        
        noise = torch.randn(image.shape[0], 3, 450, 450) * noise_level
        image_n = torch.add(image, noise)
        
        image = Variable(image).cuda()
        image_n = Variable(image_n).cuda()
        #print('Eval GPU Memory allocated: {} MB'.format(torch.cuda.memory_allocated() / 1024**2))
        v_gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        
        output = autoencoder(image_n)
        loss = loss_func(output, image)
        
        del image
        del image_n
        
        total_val_iter += 1
        total_val_loss += loss.detach().item()
        v_gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        
    
    train_loss.append(total_loss / total_iter)
    valid_loss.append(total_val_loss / total_val_iter)
    e = time.time()
    print("Iteration ", i+1)
    print('GPU Memory allocated: {} MB'.format(torch.cuda.memory_allocated() / 1024**2))
    print("Time elapsed:",e-s)