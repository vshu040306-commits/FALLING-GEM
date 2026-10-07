import cv2
import mediapipe as mp
import math
import os
import time
import serial

from collections import deque
from pythonosc.udp_client import SimpleUDPClient

# =========================================================
# 1. TouchDesigner OSC 设置
# =========================================================
TD_IP = "127.0.0.1"
TD_PORT = 8000
td_client = SimpleUDPClient(TD_IP, TD_PORT)

# =========================================================
# 2. Arduino 串口设置
# =========================================================
ARDUINO_PORT = "/dev/cu.usbmodem14101" 
ARDUINO_BAUD_RATE = 115200

# =========================================================
# 3. 摄像头设置 & 缩放控制
# =========================================================
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
current_zoom_level = 1.0  

# =========================================================
# 4. TD State 定义
# =========================================================
TD_NO_PERSON = 10
TD_PERSON_PRESENT = 11
TD_STOPPING_CUE = 0
TD_FINAL_STATE = 6

# =========================================================
# 5. Experience 时间轴
# =========================================================
STOPPING_CUE_DURATION = 5.0
EXPERIENCE_TIMELINE = [
    {"state": 1, "duration": 3.0},
    {"state": 2, "duration": 8.0},
    {"state": 3, "duration": 7.0},
    {"state": 4, "duration": 8.0},
    {"state": 5, "duration": 5.0}
]

# =========================================================
# 6. Body State 定义
# =========================================================
BODY_ACTIVE = "ACTIVE"
BODY_STABLE_HEAD_LOW = "STABLE + HEAD LOW"
BODY_STABLE_HEAD_UPRIGHT = "STABLE + HEAD UPRIGHT"
BODY_NO_PERSON = "NO PERSON"
BODY_CALIBRATING = "CALIBRATING"

# =========================================================
# 7. 身体识别参数 (修改了确认时间)
# =========================================================
MIN_VISIBILITY = 0.50         
PIXEL_JITTER_DEADZONE = 1.0   
MAX_NORM_MOVEMENT = 15.0      
SMOOTHING_FRAMES = 8          
HEAD_RATIO_SMOOTHING_FRAMES = 15

HEAD_WEIGHT = 0.30
TORSO_WEIGHT = 0.55
ARM_WEIGHT = 0.15

ACTIVE_MOVEMENT_THRESHOLD = 2.5 
STATE_CONFIRM_SECONDS = 5.0     # 【核心修改】：沉浸确认时间延长到 5 秒
STOPPING_CUE_SECONDS = 5.0
ACTIVE_RESET_SECONDS = 5.0
NO_PERSON_RESET_SECONDS = 3.0   

# =========================================================
# 8. MediaPipe 初始化
# =========================================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

TRACKED_LANDMARKS = {"nose": 0, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16}
UPPER_BODY_CONNECTIONS = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")]
HEAD_POINTS, TORSO_POINTS, ARM_POINTS = ["nose"], ["left_shoulder", "right_shoulder"], ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]

# =========================================================
# 9. 状态与缓存变量
# =========================================================
previous_positions = {}
overall_history = deque(maxlen=SMOOTHING_FRAMES)
head_ratio_history = deque(maxlen=HEAD_RATIO_SMOOTHING_FRAMES)

confirmed_body_state = BODY_CALIBRATING
confirmed_body_start_time = None  # 新增：记录当前确认状态开始的时间
candidate_body_state = None
candidate_body_start_time = None
stable_state_start_time = None

cached_shoulder_width = 100.0  
calibration_start_time = None  
calibration_frames = []        
baseline_head_ratio = None     

experience_active = False
experience_elapsed = 0.0
experience_last_update_time = None
current_td_state = None
waiting_for_object = False
experience_paused = False
active_start_time = None
no_person_start_time = None

# =========================================================
# 10. 基础数学函数
# =========================================================
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def landmark_to_pixel(landmark, width, height):
    return (int(landmark.x * width), int(landmark.y * height))

def smooth_average(history):
    return sum(history) / len(history) if history else 0.0

def get_group_average(movements, points):
    values = [movements[n] for n in points if n in movements]
    return sum(values) / len(values) if values else 0.0

def calculate_weighted_movement(movements, w_head, w_torso, w_arms):
    head = get_group_average(movements, HEAD_POINTS)
    torso = get_group_average(movements, TORSO_POINTS)
    arms = get_group_average(movements, ARM_POINTS)
    return (head * w_head + torso * w_torso + arms * w_arms, head, torso, arms)

def calculate_head_ratio(positions):
    if "nose" not in positions: return None
    if "left_shoulder" in positions and "right_shoulder" in positions:
        shoulder_center_y = (positions["left_shoulder"][1] + positions["right_shoulder"][1]) / 2
        shoulder_width = calculate_distance(positions["left_shoulder"], positions["right_shoulder"])
    elif "left_shoulder" in positions:
        shoulder_center_y, shoulder_width = positions["left_shoulder"][1], cached_shoulder_width
    elif "right_shoulder" in positions:
        shoulder_center_y, shoulder_width = positions["right_shoulder"][1], cached_shoulder_width
    else:
        return None
    if shoulder_width < 20: shoulder_width = cached_shoulder_width
    return (shoulder_center_y - positions["nose"][1]) / shoulder_width

# =========================================================
# 11. 核心判定逻辑 (包含静态违规检测 & 修复沉浸倒计时中断)
# =========================================================
def determine_raw_body_state(person_detected, movement, head_ratio, current_time, positions):
    global calibration_start_time, baseline_head_ratio, calibration_frames
    
    if not person_detected:
        calibration_start_time = None 
        baseline_head_ratio = None     
        calibration_frames.clear()     
        return BODY_NO_PERSON

    # 静态姿态违规检测：手肘高度是否超过鼻子
    posture_invalid = False
    if "nose" in positions:
        nose_y = positions["nose"][1]
        if "left_elbow" in positions and positions["left_elbow"][1] < nose_y:
            posture_invalid = True
        if "right_elbow" in positions and positions["right_elbow"][1] < nose_y:
            posture_invalid = True

    if baseline_head_ratio is None:
        if calibration_start_time is None: calibration_start_time = current_time
        if (current_time - calibration_start_time) < 3.0:
            if head_ratio is not None and not posture_invalid: 
                calibration_frames.append(head_ratio)
            return BODY_CALIBRATING
        else:
            baseline_head_ratio = sum(calibration_frames) / len(calibration_frames) if len(calibration_frames) > 0 else 0.42
            
    if movement >= ACTIVE_MOVEMENT_THRESHOLD or posture_invalid: 
        return BODY_ACTIVE
        
    if head_ratio is not None and baseline_head_ratio is not None:
        if head_ratio < (baseline_head_ratio * 0.85): return BODY_STABLE_HEAD_LOW
    return BODY_STABLE_HEAD_UPRIGHT

def update_confirmed_body_state(raw_state, current_time):
    global confirmed_body_state, candidate_body_state, candidate_body_start_time, confirmed_body_start_time
    
    # 初始化状态计时器
    if confirmed_body_start_time is None:
        confirmed_body_start_time = current_time

    # 1. 瞬间打断 (ACTIVE)
    if raw_state == BODY_ACTIVE:
        if confirmed_body_state != BODY_ACTIVE:
            confirmed_body_state = BODY_ACTIVE
            confirmed_body_start_time = current_time
        candidate_body_state = None
        return
        
    # 2. 离开缓冲 (NO PERSON)
    if raw_state == BODY_NO_PERSON:
        if confirmed_body_state != BODY_NO_PERSON:
            if candidate_body_state != BODY_NO_PERSON:
                candidate_body_state = BODY_NO_PERSON
                candidate_body_start_time = current_time
            elif (current_time - candidate_body_start_time) >= NO_PERSON_RESET_SECONDS:
                confirmed_body_state = BODY_NO_PERSON
                confirmed_body_start_time = current_time
                candidate_body_state = None
        return

    # 3. 沉浸状态判定 (STABLE)
    if raw_state == confirmed_body_state:
        candidate_body_state = None
        return
        
    if raw_state != candidate_body_state:
        if candidate_body_state is None:
            # 刚开始计时
            candidate_body_state = raw_state
            candidate_body_start_time = current_time
            return
        elif raw_state in [BODY_STABLE_HEAD_UPRIGHT, BODY_STABLE_HEAD_LOW] and \
             candidate_body_state in [BODY_STABLE_HEAD_UPRIGHT, BODY_STABLE_HEAD_LOW]:
            # 【核心修复】：在端正和低头之间切换时，只更新标签，绝对不重置倒计时！
            candidate_body_state = raw_state 
        else:
            # 发生实质性的状态改变，重置时间
            candidate_body_state = raw_state
            candidate_body_start_time = current_time
            return
            
    # 检查是否成功熬过了 5 秒
    if (current_time - candidate_body_start_time) >= STATE_CONFIRM_SECONDS:
        confirmed_body_state = candidate_body_state
        confirmed_body_start_time = current_time
        candidate_body_state = None

def send_td_state(state):
    global current_td_state
    if int(state) != current_td_state:
        td_client.send_message("/state", int(state))
        current_td_state = int(state)
        print("TD STATE ->", state)

# =========================================================
# 12. 体验与重置逻辑
# =========================================================
def update_stopping_cue(current_time):
    global stable_state_start_time
    if confirmed_body_state in [BODY_STABLE_HEAD_LOW, BODY_STABLE_HEAD_UPRIGHT]:
        if stable_state_start_time is None: stable_state_start_time = current_time
        if (current_time - stable_state_start_time) >= STOPPING_CUE_SECONDS:
            stable_state_start_time = None
            return True
    else:
        stable_state_start_time = None
    return False

def start_experience(current_time):
    global experience_active, experience_elapsed, experience_last_update_time, experience_paused, active_start_time
    experience_active, experience_paused = True, False
    experience_elapsed, active_start_time = 0, None
    experience_last_update_time = current_time
    send_td_state(TD_STOPPING_CUE)

def reset_system(reason):
    global experience_active, experience_elapsed, experience_paused, active_start_time
    global waiting_for_object, stable_state_start_time, no_person_start_time
    global calibration_start_time, baseline_head_ratio, calibration_frames, confirmed_body_start_time
    
    print("RESET:", reason)
    if arduino:
        try: arduino.write("RESET_HOME\n".encode())
        except: pass
    
    experience_active, experience_paused, waiting_for_object = False, False, False
    experience_elapsed, active_start_time, stable_state_start_time, no_person_start_time = 0, None, None, None
    
    calibration_start_time, baseline_head_ratio = None, None
    calibration_frames.clear()
    previous_positions.clear()
    overall_history.clear()
    head_ratio_history.clear()
    confirmed_body_start_time = None # 重置状态计时器
    
    send_td_state(TD_NO_PERSON)

# =========================================================
# 13. 初始化硬件
# =========================================================
try:
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=ARDUINO_BAUD_RATE, timeout=0.05)
    time.sleep(2)
    print("Arduino Connected!")
except:
    arduino = None
    print("Arduino Not Found. Running in Demo Mode.")

camera = cv2.VideoCapture(CAMERA_ID)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Started: Press ] to Zoom In, [ to Zoom Out, R to Reset, Q to Quit")

# =========================================================
# 14. 主循环
# =========================================================
try:
    while True:
        ret, frame = camera.read()
        if not ret: break
            
        current_time = time.perf_counter()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        if current_zoom_level > 1.0:
            new_w = int(w / current_zoom_level)
            new_h = int(h / current_zoom_level)
            x1 = (w - new_w) // 2
            y1 = (h - new_h) // 2
            cropped = frame[y1:y1+new_h, x1:x1+new_w]
            frame = cv2.resize(cropped, (w, h))

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        current_positions = {}
        movements = {}

        if results.pose_landmarks:
            margin_x, margin_y = w * 0.02, h * 0.02 
            for name, index in TRACKED_LANDMARKS.items():
                lm = results.pose_landmarks.landmark[index]
                if lm.visibility < MIN_VISIBILITY: continue 
                point = landmark_to_pixel(lm, w, h)
                if margin_x < point[0] < w - margin_x and margin_y < point[1] < h - margin_y:
                    current_positions[name] = point
                    cv2.circle(frame, point, 5, (0, 255, 0), -1)
                    cv2.putText(frame, name, (point[0] + 8, point[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        person_detected = ("nose" in current_positions) and (("left_shoulder" in current_positions) or ("right_shoulder" in current_positions))

        if person_detected:
            xs, ys = [p[0] for p in current_positions.values()], [p[1] for p in current_positions.values()]
            if (max(xs) - min(xs)) < w * 0.05 or (max(ys) - min(ys)) < h * 0.05:
                person_detected = False
                current_positions.clear()

        if not person_detected:
            cv2.putText(frame, "NO PERSON", (int(w*0.4), int(h*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        if person_detected:
            if "left_shoulder" in current_positions and "right_shoulder" in current_positions:
                cw = calculate_distance(current_positions["left_shoulder"], current_positions["right_shoulder"])
                if cw > 20: cached_shoulder_width = cw
            else:
                cw = cached_shoulder_width
                
            for name in current_positions:
                if name in previous_positions:
                    raw_move = calculate_distance(current_positions[name], previous_positions[name])
                    if raw_move < 150.0:
                        norm_move = (max(0, raw_move - PIXEL_JITTER_DEADZONE) / cw) * 100.0
                        movements[name] = min(norm_move, MAX_NORM_MOVEMENT)

            for pA, pB in UPPER_BODY_CONNECTIONS:
                if pA in current_positions and pB in current_positions:
                    cv2.line(frame, current_positions[pA], current_positions[pB], (0, 200, 255), 3)

            previous_positions.clear()
            previous_positions.update(current_positions)

            if movements:
                t_head, t_torso, t_arms = (0.2, 0.8, 0.0) if not any(x in movements for x in ARM_POINTS) else (HEAD_WEIGHT, TORSO_WEIGHT, ARM_WEIGHT)
                total, head, torso, arms = calculate_weighted_movement(movements, t_head, t_torso, t_arms)
                overall_history.append(total)
        else:
            previous_positions.clear()

        movement = smooth_average(overall_history)
        head_ratio = calculate_head_ratio(current_positions)
        if head_ratio is not None: head_ratio_history.append(head_ratio)
        if len(head_ratio_history): head_ratio = smooth_average(head_ratio_history)

        raw_state = determine_raw_body_state(person_detected, movement, head_ratio, current_time, current_positions)
        update_confirmed_body_state(raw_state, current_time)

        # =============== 业务流转 ===============
        if not experience_active:
            if confirmed_body_state == BODY_NO_PERSON:
                send_td_state(TD_NO_PERSON)
            else:
                send_td_state(TD_PERSON_PRESENT)
                
            if update_stopping_cue(current_time): start_experience(current_time)
        else:
            if not person_detected:
                if no_person_start_time is None: no_person_start_time = current_time
                if (current_time - no_person_start_time) >= NO_PERSON_RESET_SECONDS: reset_system("Person left for 3s")
            else:
                no_person_start_time = None
                
            if confirmed_body_state == BODY_ACTIVE:
                experience_paused = True
                if active_start_time is None: active_start_time = current_time
                if (current_time - active_start_time) >= ACTIVE_RESET_SECONDS: reset_system("ACTIVE timeout")
            else:
                experience_paused, active_start_time = False, None
                
            if not waiting_for_object and not experience_paused:
                experience_elapsed += (current_time - experience_last_update_time)
                experience_last_update_time = current_time
                
                if experience_elapsed >= STOPPING_CUE_DURATION:
                    tt = experience_elapsed - STOPPING_CUE_DURATION
                    total_dur = 0
                    for item in EXPERIENCE_TIMELINE:
                        total_dur += item["duration"]
                        if tt < total_dur:
                            send_td_state(item["state"])
                            break
                    else:
                        waiting_for_object = True
                        send_td_state(TD_FINAL_STATE)
                        if arduino:
                            try: arduino.write("ENTER_STATE_6\n".encode())
                            except: pass

        if arduino:
            try:
                while arduino.in_waiting:
                    line = arduino.readline().decode().strip()
                    if line == "OBJECT_PLACED": reset_system("Object placed")
            except: pass

 # =============== UI 显示 ===============
        # 计算当前状态已持续的秒数
        state_dur = (current_time - confirmed_body_start_time) if confirmed_body_start_time else 0.0

        cv2.putText(frame, f"TD:{current_td_state}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"BODY:{confirmed_body_state} ({state_dur:.1f}s)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # --- 新增：实时显示动态标尺 (Shoulder Width) 与 归一化运动比例 (Disp/Width) ---
        cv2.putText(frame, f"SHOULDER WIDTH: {cached_shoulder_width:.1f} px", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        cv2.putText(frame, f"NORM MOVE (Disp/Width): {movement:.2f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"ZOOM:{current_zoom_level:.1f}x", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)

        # 动态 Y 轴排版，避免文字重叠
        y_offset = 220
        if baseline_head_ratio is None and person_detected:
            calib_elapsed = (current_time - calibration_start_time) if calibration_start_time else 0.0
            cv2.putText(frame, f"CALIBRATING ({calib_elapsed:.1f}s / 3.0s)", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            y_offset += 40
        elif experience_paused:
            cv2.putText(frame, "PAUSED (Movement or Bad Posture)", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 40

        # 当正在确认 5 秒沉浸时，显示进度倒计时
        if candidate_body_state in [BODY_STABLE_HEAD_UPRIGHT, BODY_STABLE_HEAD_LOW]:
            cand_elapsed = current_time - candidate_body_start_time
            cv2.putText(frame, f"WAITING FOR IMMERSION: {cand_elapsed:.1f}s / {STATE_CONFIRM_SECONDS}s", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Stopping Cue", frame)
        key = cv2.waitKey(1) & 0xff
        
        if key == ord("q"): break
        if key == ord("r"): reset_system("manual")
        if key == ord("]"): current_zoom_level = min(3.0, current_zoom_level + 0.1)
        if key == ord("["): current_zoom_level = max(1.0, current_zoom_level - 0.1)

finally:
    send_td_state(TD_NO_PERSON)
    if arduino:
        try: arduino.write("RESET_HOME\n".encode())
        except: pass
        arduino.close()
    camera.release()
    pose.close()
    cv2.destroyAllWindows()
