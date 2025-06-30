
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
from keyboard_input_only_rev00 import controller
from test_lanenet_final_rev04 import main_autonomous_loop
import pyautogui


import win32ui #'CreateDCFromHandle' 사용시 필요
import win32con #비트블릿 복사할 때 필요
import win32gui #ReleaseDC 할 때 필요
import pydirectinput as pyautogui #pydirectinput을 통해 가속(w), 감속(s), 조향(a/d) 키보드 입력을 게임에 전달.
import pydirectinput
pydirectinput.FAILSAFE = False

# ==========================
# LaneNet 기반 자율주행 실행 스크립트
# preprocess_for_lanenet.py  →  test_lanenet_final_rev04.py  →  drive_with_lanenet_control.py
# keyboard_input_only_rev00.py → (직접 import) → drive_with_lanenet_control.py
# ==========================

def calculate_steering_from_fit(fit_params, image_width=1280):
    """
    다항식 차선 좌/우 곡선(fit_params)으로부터
    이미지 중심 대비 차선 중심 오프셋을 계산
    """
    if fit_params is None or len(fit_params) < 2:
        return 0  # 차선 정보 부족 시 정중앙 유지

    left_fit, right_fit = fit_params[0], fit_params[1]

    y_eval = 720  # 이미지 하단 기준
    left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    lane_center = (left_x + right_x) / 2
    image_center = image_width / 2
    offset = lane_center - image_center

    return offset  # (+) 우측 편향, (-) 좌측 편향

def apply_control(offset, threshold=20):
    """
    오프셋 기준으로 pyautogui 키 입력 수행
    """
    pyautogui.keyDown('w')  # 항상 전진

    if offset < -threshold:
        pyautogui.keyDown('a')
        pyautogui.keyUp('d')
    elif offset > threshold:
        pyautogui.keyDown('d')
        pyautogui.keyUp('a')
    else:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')

def stop_control():
    """
    자율주행 OFF 상태일 때 모든 조작 해제
    """
    pyautogui.keyUp('w')
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')

def main_drive_loop(weights_path):
    print("[INFO] 자율주행 시작 (Y/N 전환, ESC 종료)")

    while True:
        controller.check_key_events()

        if controller.is_exit_pressed():
            print("[INFO] ESC 입력 - 자율주행 종료")
            stop_control()
            break

        if controller.is_auto_drive_enabled():
            result = main_autonomous_loop(weights_path) #main_autonomous_loop함수를 무한루프로 호출하므로 GTA5에서 차선 추출가능
            offset = calculate_steering_from_fit(result['fit_params'])
            apply_control(offset)
        else:
            stop_control()

        time.sleep(0.05)  # 과도한 반복 방지 #0.05초 간격

if __name__ == '__main__':
    # 실제 모델 체크포인트 경로 수정 필요
    #weights_path = 'E:/gta5_project/lanenet_model/tusimple_lanenet.ckpt'
    weights_path = 'E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference/lanenet_maybeshewill/tusimple_lanenet.ckpt' #이렇게 ckpt까지 지정해줘야함. 폴더명으로 하면 액세스 오류 남.
    main_drive_loop(weights_path)
