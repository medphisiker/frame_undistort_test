import camera_calib

# путь к файлу с фото, которое мы хотим выправить
distorted_img_filename = "frame_0.jpg"

# путь к файлу калибровки камеры в формате *.json или *.p для pickle'
calibration_file = "camera_calib.json"

# работа скрипта
camera_calib.undistort_frame(distorted_img_filename, calibration_file)