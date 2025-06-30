
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
from virtual_lane import draw_virtual_centerline #가상 중앙 차선 그려주는 함수
import cv2
from preprocess_for_lanenet import grab_speed_region #속도계 영역 캡쳐
import pytesseract #문자인식 모듈
# 설치한 경로를 정확히 넣어줘야 함
pytesseract.pytesseract.tesseract_cmd = r"E:/gta5_projectOCRtesseract/tesseract.exe"  # 절대경로로 지정해줘야 함


# ==========================
# LaneNet 기반 자율주행 실행 스크립트
# preprocess_for_lanenet.py  →  test_lanenet_final_rev04.py  →  drive_with_lanenet_control.py
# keyboard_input_only_rev00.py → (직접 import) → drive_with_lanenet_control.py
# ==========================


# 전역 변수로 선언 필요
frame_count = 0



#calculate_steering_from_fit() 함수는 fit_params를 기반으로 조향 오프셋을 계산
#이 fit_params는 lanenet_postprocess.py의 postprocess() 함수에서 생성
#왼쪽 차선만 감지되었을 때는 fit_params[0]만 존재
#오른쪽 차선만 감지되었을 때는 fit_params[1]만 존재


def calculate_steering_from_fit(fit_params, image_width=1280):
    if not fit_params:
        print("[WARNING] 차선을 감지하지 못함 - 정지")
        return None  # 차선 없음 → 정지 조건 유도

    height = 720  # y 좌표 기준선
    if len(fit_params) == 2:
        # 양쪽 차선 감지됨 → 두 차선 중심선 계산
        left_x = fit_params[0][0] * height ** 2 + fit_params[0][1] * height + fit_params[0][2]
        right_x = fit_params[1][0] * height ** 2 + fit_params[1][1] * height + fit_params[1][2]
        lane_center = (left_x + right_x) / 2 #양쪽 차선 중앙으로 달리기
    elif len(fit_params) == 1: #한쪽 차선만 인식될 경우
        # 한쪽 차선만 감지 → 우측 주행 기준으로 보정
        one_x = fit_params[0][0] * height ** 2 + fit_params[0][1] * height + fit_params[0][2]
        lane_center = one_x - 200  # 우측 차선 기준이면 +200 / 좌측이면 -200으로 offset 조절
                                   #-200 또는 +200은 상황에 따라 조정할 수 있는 휴리스틱 값
    else:
        return None

    image_center = image_width / 2
    offset = lane_center - image_center  # 화면 중심과 차선 중심의 차이
    return offset


#화면에서 속도 추출
def get_current_speed_from_screen():
    region = grab_speed_region()
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)  # 시각적으로 잘 보이게 전처리
    text = pytesseract.image_to_string(thresh, config='--psm 7 digits')

    try:
        speed = int(''.join(filter(str.isdigit, text)))
        return speed
    except:
        return None




#60km/h 속도 제한, w점진적 누르기, steering도 천천히 
def apply_control(offset, threshold=10, slow_down_zone=20, max_speed_kmh=60):
    global frame_count
    frame_count += 1

    if offset is None:
        pyautogui.keyUp('w')
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
        return

    offset = np.clip(offset, -100, 100)

    # 현재 속도 가져오기
    current_speed = get_current_speed_from_screen()
    print(f"[INFO] 현재 속도: {current_speed} km/h") #콘솔에 [INFO] 현재 속도 표시

    # 속도 기반 점진적 가속 제어
    if current_speed < max_speed_kmh and abs(offset) < slow_down_zone: #	abs(offset)가 slow_down_zone보다 크면 W 중지 --> 중심 벗어날때 감속로직
        # 프레임 기준으로 W를 조금씩 누르기 (점진적 가속)
        if frame_count % 10 == 0: 
            pyautogui.keyDown('w')  #	frame_count를 기준으로 W를 짧게 눌렀다 떼기를 반복
        elif frame_count % 10 == 3:
            pyautogui.keyUp('w')
    else:
        # 속도가 너무 빠르거나 차선 중심에서 멀면 가속 중지
        pyautogui.keyUp('w')

    # 조향 제어
    if offset < -threshold:
        pyautogui.keyDown('a')
        pyautogui.keyUp('d')
        time.sleep(0.05) #time.sleep(0.05)로 좌우 조향 천천히
    elif offset > threshold:
        pyautogui.keyDown('d')
        pyautogui.keyUp('a')
        time.sleep(0.05) #time.sleep(0.05)로 좌우 조향 천천히
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
            result = main_autonomous_loop(weights_path)  # Lanenet 실행 → 차선 추출
            offset = calculate_steering_from_fit(result.get('fit_params'))

            # ⬇ 가상 중앙선 시각화
            if result.get('source_image') is not None and offset is not None:
                result_image = draw_virtual_centerline(result['source_image'], offset) #가상 중앙차선 시각화, source_image는 grab_screen에서 캡쳐한 원본이미지, 가상 중앙선 시각화에 사용
                cv2.imshow("virtual_center", result_image)

            # ⬇ 제어 적용 (source_image도 함께 전달 가능)
            #apply_control(offset, source_image=result.get('source_image'))  --> 잘못된 호출 , apply control에는 source image 불필요
            apply_control(offset)

        else:
            stop_control()

        time.sleep(0.05)  # 반복 속도 제한









if __name__ == '__main__':
    # 실제 모델 체크포인트 경로 수정 필요
    #weights_path = 'E:/gta5_project/lanenet_model/tusimple_lanenet.ckpt'
    weights_path = 'E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference/lanenet_maybeshewill/tusimple_lanenet.ckpt' #이렇게 ckpt까지 지정해줘야함. 폴더명으로 하면 액세스 오류 남.
    main_drive_loop(weights_path)
    # 주행 루프 내에서 예시
 