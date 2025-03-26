import random
from scipy.ndimage import rotate
import numpy as np
import torch.nn.functional as F
import torch


def add_gaussian_noise(image: np.array, mean=0, std=3276):
    """
    Add Gaussian noise to the image with given mean and standard deviation.
    
    Args:
        image (np.array): The image to add noise to
        mean (float): The mean of the Gaussian noise
        std (float): The standard deviation of the Gaussian noise
    
    Returns:
        np.array: The image with added Gaussian noise
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise

    return noisy_image


def augment_rotation(image: np.array): 
    rotation_axis = random.choice([0,1,2])
    rotation_angle = random.choice(range(-15, 16))

    np_img = rotate(image, rotation_angle, axes=(rotation_axis, (rotation_axis + 1) % 3), reshape=False)

    return np_img


def load_with_augment(image_path: str):
    image = np.load(image_path)

    if random.random() < 0.5:
        return image 
    else:
        organ = image_path.split('/')[-2]
        num_of_remove_slices = random.choice(range(10,21))
        if organ == 'chest':
            num_of_remove_slices_2 = random.choice(range(1,21))
            image = image[num_of_remove_slices:]
            image = image[:-num_of_remove_slices_2]
            # if image.shape[0] == 0: 
            #     with open('error.txt', 'a') as f:
            #         f.write(f"{num_of_remove_slices}_{num_of_remove_slices_2}_{image_path}\n")
            #     print(f"{num_of_remove_slices}_{num_of_remove_slices_2}_{image_path}\n")
        elif organ == 'abdomen_pelvis':
            image = image[num_of_remove_slices:]
        elif organ == 'head_neck':
            image = image[:-num_of_remove_slices]
        else:
            raise ValueError(f"Invalid organ: {organ}")
    
    if random.random() < 0.5:
        return image
    else:
        return augment_rotation(image)
    

def process_image(image: np.ndarray, fix_depth=140):
    """
    Process the image from D x H x W to C x H x W x D
    - Resize the depth dimension to fix_depth using interpolation
    - Ensure fix_depth is divisible by 4 (pad if necessary)
    - Normalize pixel values by dividing by 32767
    - Convert image to (1, H, W, D) format
    
    Args:
        image (np.ndarray): The image with shape (D, H, W)
        fix_depth (int): The desired depth size
    
    Returns:
        torch.Tensor: Processed image with shape (1, H, W, D)
    """
    D, H, W = image.shape

    # Convert to torch tensor and normalize to [0, 1]
    image_tensor = torch.tensor(image, dtype=torch.float32) / 32767.0

    # Reshape to (1, 1, D, H, W) for interpolation (N, C, D, H, W)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    # Resize depth dimension using trilinear interpolation (D → fix_depth)
    image_tensor = F.interpolate(image_tensor, size=(fix_depth, 480, 480), mode='trilinear', align_corners=False)

    # Remove batch & channel dimensions → (fix_depth, H, W)
    image_tensor = image_tensor.squeeze(0).squeeze(0)

    # Ensure fix_depth is divisible by 4 (add padding if necessary)
    # pad_d = (4 - (fix_depth % 4)) % 4  # Compute required padding
    # if pad_d > 0:
    #     image_tensor = F.pad(image_tensor, (0, 0, 0, 0, 0, pad_d), mode='constant', value=0)

    # Convert (D, H, W) → (H, W, D)
    # image_tensor = image_tensor.permute(1, 2, 0)

    # Add channel dimension (1, H, W, D)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

