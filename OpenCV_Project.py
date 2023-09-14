import mediapipe as mp
import cv2
import numpy as np
import time

# Constants
ml = 100
max_x, max_y = 250 + ml, 50
curr_tool = "select tool"
col_tool = "black"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0
color = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]
color_index = 0
selected_color = color[color_index]

def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    else:
        return "erase"

def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True
    return False

hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# 도구툴
tools = cv2.imread("tools2.png")
tools = tools.astype('uint8')


# 색상 툴
def brush_tool():
    red_brush = cv2.imread("red_brush.png", cv2.IMREAD_UNCHANGED)
    yellow_brush = cv2.imread("yellow_brush.png", cv2.IMREAD_UNCHANGED)
    green_brush = cv2.imread("green_brush.png", cv2.IMREAD_UNCHANGED)
    blue_brush = cv2.imread("blue_brush.png", cv2.IMREAD_UNCHANGED)
    black_brush = cv2.imread("black_brush.png", cv2.IMREAD_UNCHANGED)
    red_brush = cv2.resize(red_brush, (50, 50))
    yellow_brush = cv2.resize(yellow_brush, (50, 50))
    green_brush = cv2.resize(green_brush, (50, 50))
    blue_brush = cv2.resize(blue_brush, (50, 50))
    black_brush = cv2.resize(black_brush, (50, 50))
    
    brush_list = [red_brush, yellow_brush, green_brush, blue_brush, black_brush]
    for i, items in enumerate(brush_list):
        brush_mask = items[:, :, 3]
        items = items[:, :, :-1]
        crop = frm[:max_y, max_x + (i * 50):max_x + 50 + (i * 50)]
        cv2.copyTo(items, brush_mask, crop)
        
def brushTool(x):
    if x < 50 + max_x:
        return "red"
    elif x < 100 + max_x:
        return "yellow"
    elif x < 150 + max_x:
        return "green"
    elif x < 200 + max_x:
        return "blue"
    else:
        return "black"

def change_color():
    global selected_color
    if col_tool == "red":
        selected_color = color[0]
    elif col_tool == "yellow":
        selected_color = color[1]
    elif col_tool == "green":
        selected_color = color[2]
    elif col_tool == "blue":
        selected_color = color[3]
    else:
        selected_color = color[4]
        

mask = np.ones((480, 640, 3), dtype=np.uint8) * 255
mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC 키를 누르면 종료
        cv2.destroyAllWindows()
        cap.release()
        break
        
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)
            change_color()
			
            # 그리기 툴
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("your current tool set to : ", curr_tool)
                    time_init = True
                    rad = 40
            # 브러쉬 색상 툴
            elif (x > max_x and x < max_x + 250) and y < max_y:
                if time_init:
                    ctime = time.time()
                    time_init = False
                btime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (btime - ctime) > 0.8:
                    col_tool = brushTool(x)
                    time_init = True
                    rad = 40
            # 클리어
            elif (x > 0 and x < 50) and  (y < 55 and y > 5):
                if time_init:
                    ctime = time.time()
                    time_init = False
                btime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (btime - ctime) > 0.8:
                    mask = np.ones((480, 640, 3), dtype=np.uint8) * 255
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    time_init = True
                    rad = 40
                    
            # 굵기        
            else:
                if (x > 50 and x < 100) and (y < 55 and y > 5):
                    if time_init:
                        ctime = time.time()
                        time_init = False
                    ptime = time.time()
                    
                    cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                    rad -= 1
                    if (ptime - ctime) > 0.8:
                        thick = (thick + 2) % 9
                        if thick == 0 or thick == 1:
                            thick = 2
                        time_init = True
                        rad = 40
                        


                
            if curr_tool == "draw":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), selected_color, thick)
                    cv2.line(frm, (prevx, prevy), (x, y), selected_color, thick)  # Copy to frm
                    prevx, prevy = x, y

                else:
                    prevx = x
                    prevy = y

            elif curr_tool == "line":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not (var_inits):
                        xii, yii = x, y
                        var_inits = True

                    cv2.line(frm, (xii, yii), (x, y), (50, 152, 255), thick)

                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), selected_color, thick)
                        cv2.line(frm, (xii, yii), (x, y), selected_color, thick)  # Copy to frm
                        var_inits = False

            elif curr_tool == "rectangle":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not (var_inits):
                        xii, yii = x, y
                        var_inits = True

                    cv2.rectangle(frm, (xii, yii), (x, y), (0, 255, 255), thick)

                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), selected_color, thick)
                        cv2.rectangle(frm, (xii, yii), (x, y), selected_color, thick)  # Copy to frm
                        var_inits = False

            elif curr_tool == "circle":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not (var_inits):
                        xii, yii = x, y
                        var_inits = True

                    cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (255, 255, 0), thick)

                else:
                    if var_inits:
                        cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), selected_color,
                                   thick)
                        cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), selected_color,
                                   thick)  # Copy to frm
                        var_inits = False

            elif curr_tool == "erase":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)
                
                if index_raised(yi, y9):
                    cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                    cv2.circle(mask, (x, y), 30, (255, 255, 255), -1)

                

    op = cv2.bitwise_and(frm, frm, mask=mask_gray)
    frm[:, :, 1] = op[:, :, 1]
    frm[:, :, 2] = op[:, :, 2]

    frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, op[:max_y, ml:max_x], 0.3, 0)
    
    clear_image = cv2.imread('clear.png',cv2.IMREAD_UNCHANGED)
    clear_image = cv2.resize(clear_image,(50,50))
    clear_mask = clear_image[:, :, 3]
    clear_image = clear_image[:, :, :-1]
    crop = frm[5:55, 0:50]
    cv2.copyTo(clear_image, clear_mask, crop)
    # border Image
    border_image = cv2.imread('border_image.png', cv2.IMREAD_UNCHANGED)
    border_image = cv2.resize(border_image, (50, 50))
    border_mask = border_image[:, :, 3]
    border_image = border_image[:, :, :-1]
    crop = frm[5:55, 50:100]
    cv2.copyTo(border_image, border_mask, crop)
    
    brush_tool()
    
    cv2.putText(frm, curr_tool, (340 + ml, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frm, "Border: " + str(thick), (340 + ml, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frm, "Color: " + col_tool, (340 + ml, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("paint app", frm)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
