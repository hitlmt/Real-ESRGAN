import os 
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
import time


def main() -> int:
    scale = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)
    for i, image in enumerate(os.listdir("inputs")):
        image = Image.open(f"inputs/{image}").convert('RGB')
        t0 = time.time()
        sr_image = model.predict(image)
        t = time.time() - t0
        print(f"### {t}s")
        sr_image.save(f'results/{image}_{scale}.png')


if __name__ == '__main__':
>>>>>>> 9327820c003a9a855e22671c3e3342fe31124c38
    main()
