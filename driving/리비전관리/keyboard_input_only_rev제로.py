# keyboard_input_only_rev00.py

#자율주행 on/off만 담당함

#import pygame
from data_collection.key_cap import key_check

# 초기화
pygame.init()
pygame.display.set_mode((100, 100))  # 창을 보여줄 필요는 없지만 이벤트 처리를 위해 필요

class AutonomousControl:
    def __init__(self):
        self.auto_drive = False
        #pygame.display.set_mode((1280, 720)) #이 가상의 창을 클릭해야 Y/N가 감지됨,#이 위치에 두면 AutonomousControl() 클래스가 생성될 때마다 창이 정확히 만들어지고, Y/N/ESC 입력이 안정적으로 들어오게 됨
        #pygame.display.set_caption("AI Autodrive Control") 

    def check_key_events(self):
        """
        키보드 이벤트를 감지하여 자율주행 상태를 업데이트
        """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    self.auto_drive = True
                    print("[KEYBOARD] 자율주행 ON (Y키 입력)")
                elif event.key == pygame.K_n:
                    self.auto_drive = False
                    print("[KEYBOARD] 자율주행 OFF (N키 입력)")

    def is_auto_drive_enabled(self):
        """
        자율주행 상태 반환
        """
        return self.auto_drive

    #esc기능 추가
    def is_exit_pressed(self): 
        keys = pygame.key.get_pressed()
        return keys[pygame.K_ESCAPE]

    


# 외부에서 쓸 수 있도록 인스턴스 생성
controller = AutonomousControl()
