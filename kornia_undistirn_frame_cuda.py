import kornia
from PIL import Image
import numpy as np
import camera_calib
import torch
import time

# Функция для сохранения torch-массива как изображения на диск:
def save_tensor_as_image(path, tensor):
    image = kornia.utils.tensor_to_image(tensor)
    im = Image.fromarray(np.uint8(image * 255))
    im.save(path)


if __name__ == "__main__":
    # путь к файлу с фото, которое мы хотим выправить
    distorted_img_filename = "frame_0.jpg"

    # путь к файлу калибровки камеры в формате *.json или *.p для pickle'
    calibration_file = "camera_calib.json"

    mtx, dist_coeff = camera_calib.read_camera_calibration(calibration_file)
    mtx = torch.tensor(mtx).unsqueeze(0).type('torch.FloatTensor').to('cuda')
    dist_coeff = torch.tensor(dist_coeff).type('torch.FloatTensor').to('cuda')

    img_bgr_tensor = kornia.io.load_image(
        distorted_img_filename, kornia.io.ImageLoadType.RGB32, device="cuda"
    )
    
    img_bgr_tensor = img_bgr_tensor.unsqueeze(0)    
    
    for i in range(0, 100):
        start = time.time()
        out = kornia.geometry.calibration.undistort_image(img_bgr_tensor, mtx, dist_coeff)
        end = time.time()
    
        print(1 / (end - start))
    
    # save_tensor_as_image("kornia_undistornt_test.jpg", out)
