#15分上限（10分）
# 改成单人game 分数板       张逸开加油(ZYK)
# 修改work by 张逸开
# 球速变化 提高下限        （留给张逸开构思） 
# 同时 n个 奖励块（3，4）   目前只有一个(CXY)


# music改到手势 ： 两根手指触碰或者别的什么  设计手势(ZYK)（音乐开关，切歌）


# 奖励块：双板长度变化（cxy），哈哈镜（zyk），生成小球，全局模糊（zyk）  surperise模块

#小球碰撞（动量守恒）（ZYK）

from vcam import vcam, meshGen
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np
import mediapipe as mp
from gesture_judgment import detect_all_finger_state, detect_hand_state
from music import Music
from ball import Ball, detect_collision, resolve_collision
import random

def haha_mirror():
    c1= vcam(720, 1280)  # 初始化虚拟镜子
    plane = meshGen(720, 1280)
    plane.Z += 20 * np.exp(-0.5 * ((plane.Y * 1.0 / plane.H) / 0.1) ** 2) / (0.1 * np.sqrt(2 * np.pi))
    pts3d = plane.getPlane()
    pts2d = c1.project(pts3d)
    map_x, map_y = c1.getMaps(pts2d)
    return 1280 - map_x,map_y

def blur(img):
    return cv2.blur(img, (20, 20))

class Game:
    def __init__(self):
        """
        初始化游戏相关的各种参数、资源以及相关模块实例
        """
        # 初始化mediapipe的手部检测相关实例
        self.mp_hands = mp.solutions.hands
        self.hands1 = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # 设置宽度
        self.cap.set(4, 720)  # 设置高度

        # 存储最近30次的手势判断结果
        self.recent_states = [''] * 30

        # 加载资源文件
        self.imgDesk = cv2.imread('images/desk.png')
        self.imgBlock1 = cv2.imread('images/block1.png', cv2.IMREAD_UNCHANGED)
        self.imgBlock2 = cv2.imread('images/block2.png', cv2.IMREAD_UNCHANGED)
        self.surpriseblock=cv2.imread('images/star.png', cv2.IMREAD_UNCHANGED)

        # 调整图像大小
        self.imgDesk = cv2.resize(self.imgDesk, dsize=(1280, 720))
        self.imgBlock1 = cv2.resize(self.imgBlock1, dsize=(50, 250))
        self.imgBlock2 = cv2.resize(self.imgBlock2, dsize=(50, 250))

        self.surpriseblock = cv2.resize(self.surpriseblock, dsize=(50, 50))


        # 触发替换背景的分数阈值
        self.score_threshold = 200

        # 手势检测参数
        self.detector = HandDetector(detectionCon=0.8, maxHands=2)
        self.init_screen()

    def init_screen(self):
        self.game_state = "READY" 
        self.balls = [Ball([200, 150], 15, 15,time.time(),1)]  # 初始只有一个小球
        self.score = 0
        self.cooldown = 0.8
        self.M=Music()
        self.M.toggle_music()
        self.last_left_pos = (0, 0)
        self.last_right_pos = (0, 0)



        self.surprise=[(random.randint(150, 1130), random.randint(100, 620)),
                       (random.randint(150, 1130), random.randint(100, 620))]


        self.cool_time = time.time() #切换歌曲的冷静时间戳
        self.map_x, self.map_y = haha_mirror() #哈哈镜的初始化
        self.haha = None #当哈哈镜不启动时候处于None，启动时候变成时间戳
        self.blur = None #模糊处理不启动时候处于None，启动时候变成时间戳

        self.long_block=time.time()
        self.short_block=time.time()

        self.long_block_tag=False
        self.short_block_tag=False

        self.scoresp=True

    def longblock(self):
        self.imgBlock1 = cv2.resize(self.imgBlock1, dsize=(50, 500))
        self.imgBlock2 = cv2.resize(self.imgBlock2, dsize=(50, 500))
        self.long_block=time.time()
        self.long_block_tag=True
    def shortblock(self):
        self.imgBlock1 = cv2.resize(self.imgBlock1, dsize=(50, 150))
        self.imgBlock2 = cv2.resize(self.imgBlock2, dsize=(50, 150))
        self.short_block=time.time()
        self.short_block_tag=True


    def surperise(self,img,ball):
        for suppos in self.surprise:
            img = cvzone.overlayPNG(img, self.surpriseblock, suppos)

            if ball.pos[0]>suppos[0]-30 and ball.pos[1]>suppos[1]-30 and ball.pos[0]<suppos[0]+50+30 and ball.pos[1]<suppos[1]+50+30:

                self.surprise.remove(suppos)
                self.surprise.append((random.randint(150, 1130), random.randint(100, 620)))

                lucky=random.randint(1,7)
                self.score+=1

                if lucky==1:
                    #板长变长
                    self.longblock()
                elif lucky==2:
                    #板长变短
                    self.shortblock()
                elif lucky==3:
                    #球增多
                    self.spawn_balls()
                    self.spawn_balls()
                # elif lucky==4:
                #     #球减少
                #     self.delete_balls()
                elif lucky==4:
                    #哈哈镜效果
                    self.haha = time.time()
                elif lucky==5:
                    ball.speed_up()
                elif lucky==6:
                    self.blur = time.time()




        current=time.time()
        if current-self.long_block>5:
            self.long_block_tag=False
            if self.short_block_tag==False:
                self.imgBlock1 = cv2.resize(self.imgBlock1, dsize=(50, 250))
                self.imgBlock2 = cv2.resize(self.imgBlock2, dsize=(50, 250))

        if current-self.short_block>5:
            self.short_block_tag=False
            if self.long_block_tag==False:
                self.imgBlock1 = cv2.resize(self.imgBlock1, dsize=(50, 250))
                self.imgBlock2 = cv2.resize(self.imgBlock2, dsize=(50, 250))
        



    def edge_detection(self, img):

        """
        对输入图像进行边缘检测效果处理（暂停时使用）

        :param img: 输入的图像
        :return: 经过边缘检测处理后的图像
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    def spawn_balls(self):
        if len(self.balls) == 5:
            pass
        else:
            typical=[1,2,3,4,5]
            count=[]
            for ball in self.balls:
                count.append(ball.type)
            for t in count:
                if t in typical:
                    typical.remove(t)

            self.balls.append(Ball([random.randint(300, 900), random.randint(150, 450)], 15, 15,time.time(),typical[0]))

    def delete_balls(self):
        if len(self.balls) > 1:
            self.balls.pop(-1)
    def show_ready_screen(self, img):
        """
        在图像上显示准备界面的相关提示信息

        :param img: 要显示提示信息的图像
        """
        cvzone.putTextRect(img, "Ready to Play!", (400, 320), scale=5, thickness=2,colorT=(0, 0, 0),   # 设置文本颜色为灰色
                   colorR=(230,230, 230))


    def detect_gesture(self, frame):
        """
        检测手势，根据检测结果更新游戏状态等相关变量

        :param frame: 当前帧图像
        :return: 当前手势状态
        """
        frame = cv2.flip(frame, 1)  # 水平镜像翻转
        h, w = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = self.hands1.process(image)
        current_state = "None"
        foever_state = "None"

        if keypoints.multi_hand_landmarks:
            lm = keypoints.multi_hand_landmarks[0]
            lmHand = self.mp_hands.HandLandmark

            landmark_list = [[] for _ in range(6)]  # landmark_list有6个子列表，分别存储着根节点坐标（0点）以及其它5根手指的关键点坐标

            for index, landmark in enumerate(lm.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                if index == lmHand.WRIST:
                    landmark_list[0].append((x, y))
                elif 1 <= index <= 4:
                    landmark_list[1].append((x, y))
                elif 5 <= index <= 8:
                    landmark_list[2].append((x, y))
                elif 9 <= index <= 12:
                    landmark_list[3].append((x, y))
                elif 13 <= index <= 16:
                    landmark_list[4].append((x, y))
                elif 17 <= index <= 20:
                    landmark_list[5].append((x, y))

            # 将所有关键点的坐标存储到一起，简化后续函数的参数
            all_points = {'point0': landmark_list[0][0],
                          'point1': landmark_list[1][0], 'point2': landmark_list[1][1], 'point3': landmark_list[1][2],
                          'point4': landmark_list[1][3],
                          'point5': landmark_list[2][0], 'point6': landmark_list[2][1], 'point7': landmark_list[2][2],
                          'point8': landmark_list[2][3],
                          'point9': landmark_list[3][0], 'point10': landmark_list[3][1], 'point11': landmark_list[3][2],
                          'point12': landmark_list[3][3],
                          'point13': landmark_list[4][0], 'point14': landmark_list[4][1], 'point15': landmark_list[4][2],
                          'point16': landmark_list[4][3],
                          'point17': landmark_list[5][0], 'point18': landmark_list[5][1], 'point19': landmark_list[5][2],
                          'point20': landmark_list[5][3]}

            # 调用函数，判断每根手指的弯曲或伸直状态
            bend_states, straighten_states = detect_all_finger_state(all_points)

            # 调用函数，检测当前手势
            current_state = detect_hand_state(all_points, bend_states, straighten_states)
            # 更新最近状态列表
            self.recent_states.pop(0)
            self.recent_states.append(current_state)

            for hand_landmarks in keypoints.multi_hand_landmarks:  # keypoints.multi_hand_landmarks是一个列表，只有一个元素，存储着21个关键点的xyz坐标。使用for循环遍历
                foever_state = current_state

        if foever_state == "Like" and self.game_state == "READY":
            self.game_state = "RUNNING"
        if foever_state == "Return" and self.game_state == "RUNNING":
            self.game_state = "PAUSE"
            # self.paused = True
        if foever_state == "OK" and self.game_state == "PAUSE":
            self.game_state = "RUNNING"
            # self.paused = False
        if foever_state == "Pause" and self.game_state == "PAUSE" and time.time() - self.cool_time >= 3:
            self.cool_time = time.time()
            self.M.switch_music()
        return current_state

    def calculate_paddle_speed(self, current_paddle_pos, last_paddle_pos):
        """
        计算拍子（球拍）的移动速度
        :param current_paddle_pos: 当前拍子位置
        :param last_paddle_pos: 上次拍子位置
        :return: 拍子在x和y方向的移动速度
        """
        paddle_speed_x = current_paddle_pos[0] - last_paddle_pos[0]
        paddle_speed_y = current_paddle_pos[1] - last_paddle_pos[1]
        return paddle_speed_x, paddle_speed_y

    def run_game_loop(self):
        """
        游戏主循环，处理游戏的各种逻辑，包括读取摄像头帧、手势检测、游戏状态更新、画面渲染等
        """
        while True:
            success, frame = self.cap.read()  # 读取一帧
            img = cv2.flip(frame, flipCode=1)  # 镜像翻转图像
            hands, img = self.detector.findHands(img, flipType=False)
            if self.haha:
                if time.time() - self.haha <= 5:
                    img = cv2.remap(img, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR)  # 哈哈镜特效
                else:
                    self.haha = None
            # if self.blur:
            #     if time.time() - self.blur <= 5:
            #         img = blur(img)
            #     else:
            #         self.blur = None

            # 游戏准备界面
            if self.game_state == "READY":
                self.show_ready_screen(img)

            # 游戏运行状态
            elif self.game_state == "RUNNING":
                # 根据分数判断是否使用自定义背景
                if self.score >= self.score_threshold :
                    pass
                else:
                    # 原游戏中的碰撞检测、得分等逻辑

                    img = cv2.addWeighted(img, 0.4, self.imgDesk, 0.6, 0)  # 使用OpenCV的addWeighted

                for ball in self.balls:

                    ball.speedlimit()
                    
                    self.surperise(img, ball)

                    # if self.haha:
                    #     if time.time() - self.haha <= 3:
                    #         img = cv2.remap(img, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR)  # 哈哈镜特效
                    #     else:
                    #         self.haha = None

                    if hands:
                        # 处理手部输入来控制球拍和碰撞检测
                        for hand in hands:
                            x, y, z = hand['lmList'][8]  # 获取食指的坐标
                            h1, w1 = self.imgBlock1.shape[0:2]
                            y1 = y - h1 // 2  # 球拍y坐标，随着手部移动

                            # 左手控制左球拍
                            if hand['type'] == 'Left':
                                self.current_left_pos = (80, y1)
                                img = cvzone.overlayPNG(img, self.imgBlock1, self.current_left_pos)
                                left_speed_x, left_speed_y = self.calculate_paddle_speed(self.current_left_pos, self.last_left_pos)
                                self.last_left_pos = self.current_left_pos
                                if 100 < ball.pos[0] < 100 + w1 and y1 < ball.pos[1] < y1 + h1:
                                    current_time = time.time()
                                    if current_time - ball.lastx > self.cooldown:
                                        ball.opposite_x(left_speed_x, left_speed_y)
                                        ball.lastx = current_time
                                        self.score += 1
                                        # self.spawn_balls()
                                # 右手控制右球拍
                            if hand['type'] == 'Right':
                                self.current_right_pos = (1150, y1)
                                img = cvzone.overlayPNG(img, self.imgBlock2, self.current_right_pos)
                                right_speed_x, right_speed_y = self.calculate_paddle_speed(self.current_right_pos, self.last_right_pos)
                                self.last_right_pos = self.current_right_pos
                                if 1100 < ball.pos[0] < 1100 + w1 and y1 < ball.pos[1] < y1 + h1:
                                    current_time = time.time()
                                    if current_time - ball.lastx > self.cooldown:
                                        ball.opposite_x(right_speed_x, right_speed_y)
                                        ball.lastx = current_time
                                        self.score += 1

                    # 球的运动和边界检测
                    if ball.pos[0] < 30 or ball.pos[0] > 1150:
                        self.balls.remove(ball)
                        if len(self.balls)==0:
                            self.game_state = "READY"  # 如果球越过边界，游戏结束
                            self.init_screen()
                    else:
                        if ball.pos[1] >= 600 or ball.pos[1] <= 50:
                            current_time = time.time()
                            if current_time - ball.lasty > self.cooldown:
                                ball.opposite_y()
                                ball.lasty = current_time


                    # 游戏结束后显示分数
                    cvzone.putTextRect(img, f'score: {self.score}', (540, 710),colorT=(0, 0, 0),   # 设置文本颜色为灰色
                   colorR=(230,230, 230))
                    ball.move()


                    # 添加球和球桌（如果未使用自定义背景的情况）
                    if self.score < self.score_threshold :
                        img = cvzone.overlayPNG(img, ball.imgBall, ball.pos)


                    # if self.haha:
                    #     if time.time() - self.haha <= 3:
                    #         img = cv2.remap(img, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR)  # 哈哈镜特效
                    #     else:
                    #         self.haha = None



                current_time1 = time.time()  # 获取当前时间的时间戳
                for i in range(len(self.balls)):    # 新增：球与球之间的碰撞检测及处理
                    for j in range(i+1, len(self.balls)):
                        ball1 = self.balls[i]
                        ball2 = self.balls[j]

                        # 检查球1与球2之间的上次碰撞时间是否在3秒以内
                        # 如果小于3秒，则跳过碰撞处理
                        last_collision_time_1 = ball1.last_collision_times.get(id(ball2), 0)
                        last_collision_time_2 = ball2.last_collision_times.get(id(ball1), 0)

                        if current_time1 - last_collision_time_1 < 2 or current_time1 - last_collision_time_2 < 2:
                            continue  # 如果这两个球最近碰撞过，则跳过

                        # 如果发生碰撞，处理碰撞并更新时间戳
                        if detect_collision(ball1, ball2):
                            self.balls[i], self.balls[j] = resolve_collision(ball1, ball2)

                            # 更新球1与球2之间的碰撞时间戳
                            ball1.last_collision_times[id(ball2)] = current_time1
                            ball2.last_collision_times[id(ball1)] = current_time1

            # 暂停时加入炫酷黑白效果
            if self.game_state == "PAUSE":
                img = self.edge_detection(img)  # 使用边缘检测效果


            # 检测手势来启动、暂停、继续游戏
            self.detect_gesture(frame)

            if self.score%5==1 and self.scoresp==True:
                self.spawn_balls()
                self.scoresp=False
            if self.score%5==2:
                self.scoresp=True

            # 如果游戏退出
            if self.game_state == "EXIT":
                break
            if self.blur:
                if time.time() - self.blur <= 5:
                    img = blur(img)
                else:
                    self.blur = None
            # 显示图像
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC退出
                break

            for ball in self.balls:
                print(ball.speedx,ball.speedy)


        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    game = Game()
    game.run_game_loop()