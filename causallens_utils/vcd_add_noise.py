import torch
from PIL import Image

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd


def pad_to_square(img, background_color=None):
    """
    Pad the given image to a square shape.
    
    :param img: input PIL image object
    :param background_color: background fill color, defaults to white for RGB/RGBA mode, 255 for other modes
    :return: padded square image object
    """
    width, height = img.size
    max_size = max(width, height)
    
    # Set default background color based on image mode if not specified
    if background_color is None:
        if img.mode == 'RGB':
            background_color = (255, 255, 255)  
        elif img.mode == 'RGBA':
            background_color = (255, 255, 255, 255) 
        else:
            background_color = 255  
    # Create a new square background image
    result = Image.new(img.mode, (max_size, max_size), background_color)
    # Paste the original image onto the center of the background
    result.paste(img, ((max_size - width) // 2, (max_size - height) // 2))
    return result