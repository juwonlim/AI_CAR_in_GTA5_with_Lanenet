

# reading and writing files
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식
import cv2
import numpy as np
import time
from lanenet_inference.lanenet_predict import predict_lane
from data_collection.preprocess_for_lanenet import grab_screen
import pydirectinput
import win32ui #'CreateDCFromHandle' 사용시 필요
import win32con #비트블릿 복사할 때 필요
import win32gui #ReleaseDC 할 때 필요





# LaneNet 모델 가중치 경로
#weights_path = './model/lanenet.ckpt'
#weights_path = "E:/gta5_project/AI_GTA5/lanenet_inference/lanenet_maybeshewill/tusimple_lanenet.ckpt"
weights_path = 'E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference/lanenet_maybeshewill/tusimple_lanenet.ckpt'
# 초기화
print("[INFO] LaneNet 단독 테스트 시작")


def apply_controls(steering, throttle=0.4):
    """
    조향 및 전진 제어. 
    steering: -1.0(좌) ~ 1.0(우)
    throttle: 0.0 ~ 1.0 (현재는 상수, 추후 앞차 인식과 연동 가능)
    """
    steering_sensitivity = 0.4  # 조향 감도 계수 (값이 작을수록 부드럽게 회전)
    turn_threshold = 0.1  # 조향을 무시할 최소값

    # 조향 감도 보정 적용
    adjusted_steering = steering * steering_sensitivity

    if adjusted_steering < -turn_threshold:
        pydirectinput.keyDown('a')
        pydirectinput.keyUp('d')
    elif adjusted_steering > turn_threshold:
        pydirectinput.keyDown('d')
        pydirectinput.keyUp('a')
    else:
        pydirectinput.keyUp('a')
        pydirectinput.keyUp('d')

    # 일정한 전진 (후진 없음)
    pydirectinput.keyDown('w')
    pydirectinput.keyUp('s')





autonomous_mode = False #자율주행 초기 상태

while True:

    ''' 
      # 키 입력으로 자율주행 on/off
    if cv2.waitKey(1) & 0xFF == ord('y'):
        autonomous_mode = True
        print("[INFO] 자율주행 모드 활성화")
    elif cv2.waitKey(1) & 0xFF == ord('n'):
        autonomous_mode = False
        print("[INFO] 자율주행 모드 비활성화")
    elif cv2.waitKey(1) & 0xFF == ord('z'):
        print("[INFO] 프로그램 종료")
        break
    '''

   

     # 1. 키 입력 처리 (루프 시작부에만)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('y'):
        autonomous_mode = True
        print("[INFO] 자율주행 모드 활성화")
    elif key == ord('n'):
        autonomous_mode = False
        print("[INFO] 자율주행 모드 비활성화")
    elif key == ord('z') or key == 27:  # z 또는 esc
        print("[INFO] 프로그램 종료")
        break

    # 자율주행 모드가 켜진 경우에만 조향 명령 실행
    if autonomous_mode:
        apply_controls(lanenet_steering)
   

    # 1. GTA5 화면 캡처
    screen = grab_screen()
    #screen = grab_screen(region=(0, 40, 1280, 740)) #자동으로 캡쳐하므로 영역설정 불필요
    screen = cv2.resize(screen, (1280, 720))
    screen_rgb = screen #preprocess_for_lanenet파일에서 RGB로 리턴되므로 이렇게만 하면됨
    #screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    # 2. LaneNet용 입력 리사이징
    resized_input = cv2.resize(screen_rgb, (512, 256))



      


    # 3. 차선 예측
    try:
        binary_mask, _ = predict_lane(resized_input, weights_path)
        # 3-1. 조향 계산을 위한 좌우 차선 중심 계산
        lane_indices = np.where(binary_mask == 255)
        if lane_indices[0].size > 0:
            x_coords = lane_indices[1]
            lane_center_x = np.mean(x_coords)
            frame_center_x = binary_mask.shape[1] / 2
            error = (lane_center_x - frame_center_x) / frame_center_x
            lanenet_steering = error * 0.5  # 조향 감도
            #binary_mask에서 흰색(255) 픽셀의 x좌표 평균을 기준으로 차선 중심을 계산하고, 화면 중심과의 차이를 바탕으로 조향값을 산출
        else:
            lanenet_steering = 0.0

        # 4. 결과 시각화
        binary_mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        binary_mask_resized = cv2.resize(binary_mask_color, (1280, 720))

        lane_viz = cv2.addWeighted(screen, 0.7, binary_mask_resized, 0.5, 0)

       

       



        # 출력
        cv2.imshow("Input to LaneNet", resized_input)

        cv2.imshow("LaneNet Mask", binary_mask)
        
        cv2.imshow("LaneNet Visualization", lane_viz)
        cv2.resizeWindow("LaneNet Visualization", 640, 480)

        print("[DEBUG] unique values:", np.unique(binary_mask))
        print("[DEBUG] steering:", lanenet_steering)

        
    except Exception as e:
          print("[ERROR] LaneNet 예측 실패:", str(e))
        

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()



