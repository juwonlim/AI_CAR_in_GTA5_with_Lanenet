

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
from data_collection.key_cap import key_check
import pydirectinput as pyautogui #pydirectinput을 통해 가속(w), 감속(s), 조향(a/d) 키보드 입력을 게임에 전달.



# LaneNet 모델 가중치 경로
#weights_path = './model/lanenet.ckpt'
weights_path = 'E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference/lanenet_maybeshewill/tusimple_lanenet.ckpt'
# 초기화
print("[INFO] LaneNet 단독 테스트 시작")





def apply_keyboard_controls(steering, throttle):
    """
    AI가 예측한 스티어링과 스로틀 값을 바탕으로 키보드 입력 전달
    - throttle: -1 ~ 1 (음수면 후진)
    - steering: -1 ~ 1 (음수면 좌회전, 양수면 우회전)
    """

    # =====================
    # 1. 전진 (W) / 후진 (S)
    # =====================
    #if throttle > 0.2:
    if throttle > 0:
        pyautogui.keyDown('w')
        pyautogui.keyUp('s')
          
    elif throttle == 0:
        pyautogui.keyUp('w')
        pyautogui.keyUp('s')

    # =====================
    # 2. 좌/우 회전 (A/D)
    # =====================
    if steering < -0.2:
        pyautogui.keyDown('a')
        pyautogui.keyUp('d')
    elif steering > 0.2:
        pyautogui.keyDown('d')
        pyautogui.keyUp('a')
    else:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
    print("Applying control - Steering:", steering, "Throttle:", throttle)






#autonomous_mode = False #자율주행 초기 상태
#lanenet_steering = 0.0  # 초기값을 루프 밖에 미리 정의



def drive( ):

    pause = True  # 루프 바깥 drive() 함수 시작 시
    frame_count = 0  # while 루프 시작 전에 선언

    while True:

        # 1. GTA5 화면 캡처
        screen = grab_screen()
        #screen = grab_screen(region=(0, 40, 1280, 740)) #자동으로 캡쳐하므로 영역설정 불필요
        screen = cv2.resize(screen, (1280, 720)) #1280ㅌ720으로 리사이즈 , 이게 주석처리 되어 있으면 lanenet_predict.py에서 아예 처리를 못하는듯함.
        screen_rgb = screen #preprocess_for_lanenet파일에서 RGB로 리턴되므로 이렇게만 하면됨
        #screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        # 2. LaneNet용 입력 리사이징
        #resized_input = cv2.resize(screen_rgb, (512, 256)) #lanenet_predict.py와 중복이긴 함, 저 파일에     image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR) 이렇게 되어있음
        resized_input = screen_rgb
       


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
                #print("[ERROR] LaneNet 예측 실패:", str(e))
                print("[ERROR] LaneNet 예측 실패: inference단계") #차선예측이 실패했을떄 출력
                lanenet_steering = 0.0

            # 4. 결과 시각화
            binary_mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            binary_mask_resized = cv2.resize(binary_mask_color, (1280, 720))

            lane_viz = cv2.addWeighted(screen, 0.7, binary_mask_resized, 0.5, 0) #OpenCV(4.1.0) ... error: (-209:Sizes of input arguments do not match) 에러 발생
                                                                            #screen.shape: (720, 1280, 3)  vs   binary_mask.shape: (256, 512) → cvtColor 후 (256, 512, 3) → resize to (1280, 720, 3)
    
            # lane_center_x와 frame_center_x는 이미 계산되어 있는 상태에서
            # 시각화를 위한 선 그리기
            #cv2.line(lane_viz, (int(lane_center_x), 0), (int(lane_center_x), lane_viz.shape[0]), (0, 255, 0), 2)  # 초록색: 차선 중심
            #cv2.line(lane_viz, (int(frame_center_x), 0), (int(frame_center_x), lane_viz.shape[0]), (0, 0, 255), 2)  # 빨강색: 화면 중심



            

        



            # 출력
            #cv2.imshow("Input to LaneNet", resized_input)
            cv2.namedWindow("Input to LaneNet", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input to LaneNet", 320, 240)
            cv2.imshow("Input to LaneNet", resized_input)

            #cv2.imshow("LaneNet Mask", binary_mask)
            cv2.namedWindow("LaneNet Mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("LaneNet Mask", 320, 240)
            cv2.imshow("LaneNet Mask", binary_mask)



            cv2.namedWindow("LaneNet Visualization", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("LaneNet Visualization", 640, 480)
            cv2.imshow("LaneNet Visualization", lane_viz)





            print("[DEBUG] unique values:", np.unique(binary_mask))
            print("[DEBUG] steering:", lanenet_steering)


            
            #controls 의 에러, (before assignment 방지를 위한 선언)
            controls = [[0.0]]  # fallback 기본값
            throttle = 0        # 안전상 기본값 설정
            
            
            #최종 키보드 제어
            if not pause:
                try:
                    apply_keyboard_controls(controls[0][0], throttle)
                except UnboundLocalError as e:
                    print(f"[FATAL] controls not defined: {e}")
                    apply_keyboard_controls(0.0, 0.0)  # 안전하게 정지
                                
                


            #키 이벤트 처리
            keys = key_check()
            if 'T' in keys:
                cv2.destroyAllWindows()
                pause = True
                print('Paused. To exit the program press Z.')
                time.sleep(0.5)
            frame_count += 1 #while not pause의 루프 끝 직전에 추가 (즉,한 루프 끝날떄마다 증가)

           
            if 'Y' in keys:
                keys = key_check()
                pause = False
                stop = False
                print('Self-Driving mode activated')
                time.sleep(1)
            elif 'N' in keys:
                pause = True
                print('Self-driving mode deactivated')
            continue #루프 처음으로 다시
            

            
        except Exception as e:
            #print("[ERROR] LaneNet 예측 실패:", str(e)) 
            print("[ERROR] LaneNet 시각화 실패 (visualization 단계):", str(e)) #예측된 결과를 시각화하려다 실패했을 때
            lanenet_steering = 0.0
            

        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()




def main():
    # 자율주행 시작 (LaneNet + YOLO 사용)
    drive()


if __name__ == '__main__':
    main()