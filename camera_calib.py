import json
import os
import pickle

import cv2
import numpy as np
import time

def undistort_frame(distorted_img_filename, path_to_calib_file, result_folder=""):
    """Функция для выправления кадра на основе матрицы камеры
    которой был сделан кадр и вектора ее дисторсий.

    Parameters
    ----------
    distorted_img_filename : str
        путь к файлу кадра, который нужно распрямить.
    path_to_calib_file : str
        путь к файлу каллибровки камеры в формате pickle (*.p) или json(*.json)
    result_folder : str, optional
        путь к папке в которую будет сохраняться результат, by default ''
        Если данной папки не существует, она будет создана.

    Raises
    ------
    ValueError
        Поднимает ошибку если path_to_calib_file не ведет к файлу с расширением
        pickle (*.p) или json(*.json)
    """

    mtx, dist = read_camera_calibration(path_to_calib_file)

    distorted_image = cv2.imread(distorted_img_filename)
    height, width = distorted_image.shape[:2]

    # Refine camera matrix
    # Returns optimal camera matrix and a rectangular region of interest
    # смотрим для подробностей "What does the getOptimalNewCameraMatrix do in OpenCV?"
    # в "полезные ссылки.txt"
    start = time.time()
    for i in range(0, 1000):
        optimal_camera_matrix, roi_0 = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (width, height), 0, (width, height)
        )

        undistorted_image = cv2.undistort(
            distorted_image, mtx, dist, None, optimal_camera_matrix
        )
    end = time.time()
    print(end - start)
    
    # Create the output file name by removing the '.jpg' part
    img_name = os.path.split(distorted_img_filename)[-1]
    img_name, img_ext = os.path.splitext(img_name)
    img_name = img_name + "_undistorted" + img_ext

    if result_folder:
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

    new_filename = os.path.join(result_folder, img_name)

    # Save the undistorted image
    cv2.imwrite(new_filename, undistorted_image)
    print(f"Undistorted image: {new_filename}")


def read_camera_calibration(path_to_calib_file):
    ext = os.path.splitext(path_to_calib_file)[-1]
    if ext == ".p":
        calib_result = pickle.load(open(path_to_calib_file, "rb"))
    elif ext == ".json":
        with open(path_to_calib_file, "r") as data:
            calib_result = json.load(data)
    else:
        raise ValueError(
            "camera_calib путь к файлу калибровки \
        камеры в формате *.json или *.p для pickle"
        )

    # восстановление параметров калибровки
    mtx = np.array(calib_result["mtx"])
    dist = np.array(calib_result["dist"])
    return mtx, dist
