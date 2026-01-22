# Plan

1. Declare custom ViT with provision for pretrained weights, LoRA
2. instantiate dataloder from the adataset argument, but only the validation or test split depending on data
3. Declare optimizer, loss function, metrics (MSE Loss, PSNR, SSIM)
4. Training Loop: For a certain number of iterations (obtained through arguments), optimize the x for which the gradient obtained matches the gradient obtained by passing a sample through the model. -- 
    Steps: a. Pass a single sample from the dataset through the model
           b. Calculate the gradient after calculating the loss using the corresponding label
           c. This gradient will serve as the ground truth for reconstructing the input.
           d. randomly initialize a sample x which will have requires_grad = True as it will be optimzied.
           e. Pass this input x_opt through the model, calculate the cosine similarity or MSE loss of the gradient obtained with respect to the gradient calculated in step b.
           f. Also add another loss, that is the cosine similarity loss between the positional embedding gradients obtained in step b. and the positional embedding gradient obtained in step e.
           g. Update the x_opt
           h. Repeat steps e, f, g for a certain number of iterations (given as argument)
5. After the training loop finishes, calculate the metrics and save the results in viz folder.