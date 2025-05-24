import torch.jit
import numpy as np
import torch

@torch.jit.script
def svd_jit(x):
    return torch.linalg.svd(x, full_matrices=False)
torch.set_num_threads(1)
device = torch.device("cpu")
def transposed_pca_whiteing(x):
    x_tensor = torch.from_numpy(x).float().to(device)
    mean = torch.mean(x_tensor, dim=0)
    x_centered = x_tensor - mean
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        x_centered = x_centered.to(torch.bfloat16).contiguous()
        Vt, S, U = svd_jit(x_centered)
    pca_inverse = S[:, None] * U
    return Vt.cpu().numpy(), pca_inverse.cpu().numpy()

def get_enhanced(z,pca_inverse,pc):
    z_squared = np.abs(z)
    partitioned = np.partition(z_squared, kth=-pc, axis=1)
    second_largest_all = partitioned[:, -pc]
    thresholds = second_largest_all
    mask = z_squared >= thresholds[:, np.newaxis]
    masked_data_all = np.where(mask, z, 0)
    data = masked_data_all @ pca_inverse
    return data
def enhance(emg,R,trans=False,pc=15):
    if trans==True:
        emg=emg.T
    emg_e = extend_all_channels(emg,R)
    pca_data, pca_inverse = transposed_pca_whiteing(emg_e)
    enhanced_data= get_enhanced(pca_data,pca_inverse,pc)
    return enhanced_data

