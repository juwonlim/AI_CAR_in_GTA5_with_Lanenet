import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import tensorflow as tf
from keyboard_input_only_rev00 import controller
from test_lanenet_final_rev06 import main_autonomous_loop


import win32ui
import win32con
import win32gui
#import pyautogui
import pydirectinput
import pydirectinput as pyautogui
pydirectinput.FAILSAFE = False

from virtual_lane import draw_virtual_centerline
import cv2
from preprocess_for_lanenet import grab_speed_region

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"E:/gta5_projectOCRtesseract/tesseract.exe"

sys.path.append(os.path.abspath("E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference"))
import lanenet_inference.lanenet_model.lanenet as lanenet
import lanenet_inference.lanenet_model.lanenet_postprocess as lanenet_postprocess


#from lanenet_inference.global_config import cfg as CFG  -->global_config사용안함
import yaml
from easydict import EasyDict as edict
#with open("E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference/config/tusimple_lanenet.yaml", 'r') as f:
    #CFG = edict(yaml.safe_load(f))
with open("E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference/config/tusimple_lanenet.yaml", 'r', encoding='utf-8') as f:
    CFG = edict(yaml.safe_load(f))




frame_count = 0

def calculate_steering_from_fit(fit_params, image_width=1280):
    if not fit_params:
        print("[WARNING] 차선을 감지하지 못함 - 정지")
        return None

    height = 720
    if len(fit_params) == 2:
        left_x = fit_params[0][0] * height ** 2 + fit_params[0][1] * height + fit_params[0][2]
        right_x = fit_params[1][0] * height ** 2 + fit_params[1][1] * height + fit_params[1][2]
        lane_center = (left_x + right_x) / 2
    elif len(fit_params) == 1:
        one_x = fit_params[0][0] * height ** 2 + fit_params[0][1] * height + fit_params[0][2]
        lane_center = one_x - 200
    else:
        return None

    image_center = image_width / 2
    return lane_center - image_center

def get_current_speed_from_screen():
    region = grab_speed_region()
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
    try:
        speed = int(''.join(filter(str.isdigit, text)))
        return speed
    except:
        return None

def apply_control(offset, threshold=10, slow_down_zone=20, max_speed_kmh=60):
    global frame_count
    frame_count += 1

    if offset is None:
        pyautogui.keyUp('w')
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
        return

    offset = np.clip(offset, -100, 100)
    current_speed = get_current_speed_from_screen()
    print(f"[INFO] 현재 속도: {current_speed} km/h")

    if current_speed is not None and current_speed < max_speed_kmh and abs(offset) < slow_down_zone:
        if frame_count % 10 == 0:
            pyautogui.keyDown('w')
        elif frame_count % 10 == 3:
            pyautogui.keyUp('w')
    else:
        pyautogui.keyUp('w')

    if offset < -threshold:
        pyautogui.keyDown('a')
        pyautogui.keyUp('d')
        time.sleep(0.05)
    elif offset > threshold:
        pyautogui.keyDown('d')
        pyautogui.keyUp('a')
        time.sleep(0.05)
    else:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')

def stop_control():
    pyautogui.keyUp('w')
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')

def main_drive_loop(weights_path):
    print("[INFO] 자율주행 시작 (Y/N 전환, ESC 종료)")

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess = tf.Session(config=sess_config)

    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess=sess, save_path=weights_path)

    while True:
        controller.check_key_events()

        if controller.is_exit_pressed():
            print("[INFO] ESC 입력 - 자율주행 종료")
            stop_control()
            break

        if controller.is_auto_drive_enabled():
            result = main_autonomous_loop(sess, net, postprocessor, input_tensor)
            offset = calculate_steering_from_fit(result['fit_params'])
            apply_control(offset)
        else:
            stop_control()

        time.sleep(0.05)

    sess.close()

if __name__ == '__main__':
    weights_path = 'E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference/lanenet_maybeshewill/tusimple_lanenet.ckpt'
    main_drive_loop(weights_path)
