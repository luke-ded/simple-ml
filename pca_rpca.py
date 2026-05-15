# !/usr/bin/env python3

# Source: PyTorch documentation used throughout
import torch, math, time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


# Source: Finding Structure in Randomness 
# https://users.cms.caltech.edu/~jtropp/reports/HMT09-Finding-Structure-TR.pdf
def randomized_pca(X, k):
    print("Running randomized PCA...")
        
    # Center X
    x_mean = torch.mean(X, axis=0)
    X = X - x_mean
    
    # Stage A
    omega = torch.randn(X.shape[1], k + 5).to(device)
    
    q = 2
    Y = X @ omega
    for i in range(q):
        Y = X @ (X.T @ Y)
    
    Q, R = torch.linalg.qr(Y)
    
    # Stage B
    B = Q.T @ X
    U, S, V_t = torch.linalg.svd(B, full_matrices=False)
    
    # Create output
    proj_matrix = V_t[:k,:].T
    
    result = X @ proj_matrix
    reconstruction = (result @ proj_matrix.T) + x_mean
    
    return result, reconstruction


# Source: MAP 4112 notes, CAP 4630 notes    
def pca(X, k):
    print("Running standard PCA...")
            
    # Center X
    x_mean = torch.mean(X, axis=0)
    X = X - x_mean
        
    U, S, V_t = torch.linalg.svd(X, full_matrices=False)
    
    # Create output 
    proj_matrix = V_t[:k,:].T
    
    result = X @ proj_matrix
    reconstruction = (result @ proj_matrix.T) + x_mean
        
    return result, reconstruction


def load_dataset():
    print("Loading dataset...")
    
    data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    dataset = datasets.ImageFolder('./db-of-faces', transform=data_transform)

    X = []
    for image, label in dataset:
        X.append(image.flatten())
    
    X = torch.stack(X).to(device)
    
    return X


def unflatten(X):
    print("Unflattening images...")
    height = 112
    width = 92
    
    images = []
    num_images = X.shape[0]
    
    for i in range(num_images):
        img = torch.unflatten(X[i], 0, (height, width))
        images.append(img)
        
    return images

# Source: Matplotlib documentation
def display_image_grid(images_dict):
    print("Displaying image grid...")
    
    num_images = len(images_dict["Raw Images"])
    images_per_page = 10
    num_pages = int(math.ceil(num_images / images_per_page))
    
    for page_idx in range(num_pages):
        if page_idx > 0:
            do_next = input("Would you like to view the next subject? (y/n) ")
            if do_next != 'y' and do_next != 'Y':
                break
        
        fig, axs = plt.subplots(7, images_per_page, figsize=(8,6))
        for col_idx in range(images_per_page):
            paginated_index = (page_idx * images_per_page) + col_idx
            if paginated_index < num_images:
                for row_idx, image_type in enumerate(images_dict):
                    img = images_dict[image_type][paginated_index]
                    img = img.detach().cpu().numpy()
                    
                    axs[row_idx, col_idx].imshow(img, cmap="gray")
                                    
                    axs[row_idx, col_idx].set_xticks([])
                    axs[row_idx, col_idx].set_yticks([])
                                    
                    if col_idx == 0:
                        axs[row_idx, col_idx].set_ylabel(image_type, ha='right', va='center', rotation=0, fontsize=10)
        
        fig.suptitle(f"Subject {page_idx}", fontsize=16, y=0.98, weight="bold")
        plt.get_current_fig_manager().set_window_title(f"Subject {page_idx}")                       
        plt.tight_layout()
        plt.show()


def display_secs_chart(time_dict):
    pca_total = num_pca = rpca_total = num_rpca = 0
    
    print("-"*40)
    print("Seconds Taken".center(40))
    print("-"*40)
    
    for pca_type, secs in time_dict.items():
        print(f"{pca_type:<15} | {secs:.4f}".center(40))
        if "RPCA" in pca_type:
            rpca_total += secs
            num_rpca += 1
        else:
            pca_total += secs
            num_pca += 1
    
    print("-"*40)      
    print(f"{'PCA Average':<15} | {pca_total/num_pca:.4f}".center(40))
    print(f"{'RPCA Average':<15} | {rpca_total/num_rpca:.4f}".center(40))
    print("-"*40)
    
    
def display_mse_chart(mse_dict):
    print("-"*40)
    print("Mean Squared Error".center(40))
    print("-"*40)
    for pca_type, mse in mse_dict.items():
        print(f"{pca_type:<15} | {mse:.4f}".center(40))
    print("-"*40)
    
    
if __name__ == "__main__":
    print(f"device = {device}")
     
    X = load_dataset()
    images_dict = {"Raw Images": unflatten(X)}
    time_dict = {}
    mse_dict = {}

        
    for k in range(4, 7):
        # Do PCA
        start_time = time.perf_counter()
        result, reconstruction = pca(X, k)
        elapsed_time = time.perf_counter() - start_time
        
        mse = torch.mean((X - reconstruction) ** 2).item()
        
        print(f"Standard PCA with k = {k} components took {elapsed_time:.4f} seconds.")
        images_dict[f"PCA, {k} PCs"] = unflatten(reconstruction)
        time_dict[f"PCA, {k} PCs"] = elapsed_time
        mse_dict[f"PCA, {k} PCs"] = mse
        
        # Do randomized PCA
        start_time = time.perf_counter()
        result, reconstruction = randomized_pca(X, k)
        elapsed_time = time.perf_counter() - start_time
        
        mse = torch.mean((X - reconstruction) ** 2).item()
        
        print(f"Randomized PCA with k = {k} components took {elapsed_time:.4f} seconds.")
        images_dict[f"RPCA, {k} PCs"] = unflatten(reconstruction)
        time_dict[f"RPCA, {k} PCs"] = elapsed_time
        mse_dict[f"RPCA, {k} PCs"] = mse
        
    display_image_grid(images_dict)
    display_secs_chart(time_dict)
    display_mse_chart(mse_dict)