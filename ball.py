import random
import math
import cv2

class Ball:
    def __init__(self, pos, speedx, speedy,last,type):
        self.pos = pos
        self.speedx = speedx
        self.speedy = speedy
        self.minx=speedx
        self.miny=speedy
        self.lastx=last
        self.lasty=last
        self.radius = 25
        self.last_collision_times = {}# 用一个字典记录与其他球的碰撞时间，键是其他球的标识符，值是碰撞时间

        self.type=type
        if self.type==1:
            self.imgBall=cv2.imread('images/ball1.png', cv2.IMREAD_UNCHANGED)
        elif self.type==2:
            self.imgBall=cv2.imread('images/ball2.png', cv2.IMREAD_UNCHANGED)
        elif self.type==3:
            self.imgBall=cv2.imread('images/ball3.png', cv2.IMREAD_UNCHANGED)
        elif self.type==4:
            self.imgBall=cv2.imread('images/ball4.png', cv2.IMREAD_UNCHANGED)
        elif self.type==5:
            self.imgBall=cv2.imread('images/ball5.png', cv2.IMREAD_UNCHANGED)
        
        self.imgBall = cv2.resize(self.imgBall, dsize=(50, 50))

    def speed_up(self):
        self.minx+=10
        self.miny+=10
        self.speedx+=10
        self.speedy+=10
    def speed_down(self):
        self.minx-=5
        self.miny-=5
        self.speedx-=5
        self.speedy-=5
        if self.speedx==0:
            self.speedx+=1
        if self.speedy==0:
            self.speedy+=1


    def speedlimit(self):
        if self.speedx>40:
            self.speedx=40
        if self.speedy>40:
            self.speedy=40

        if self.speedx>self.minx:
            self.speedx=max(int(0.97*self.speedx),self.minx)
        if self.speedy>self.miny:
            self.speedy=max(int(0.97*self.speedy),self.miny)
        if self.speedx==0:
            self.speedx+=1
        if self.speedy==0:
            self.speedy+=1
            

        
    def move(self):
        self.pos[0] += self.speedx
        self.pos[1] += self.speedy


    def opposite_y(self):
        self.speedy = -self.speedy

    def opposite_x(self,paddle_speed_x, paddle_speed_y):
        self.speedx=min(50,int(paddle_speed_x*2+self.speedx))
        self.speedy=min(50,int(paddle_speed_y*2+self.speedy))
        self.speedx = -self.speedx

def detect_collision(ball1, ball2):
    """
    检测两个球是否碰撞，考虑运动趋势（速度）
    """
    # 当前两球圆心的距离
    current_distance = math.sqrt((ball1.pos[0] - ball2.pos[0]) ** 2 + (ball1.pos[1] - ball2.pos[1]) ** 2)
    # 计算两球相对速度在x和y方向的分量
    relative_vx = ball2.speedx - ball1.speedx
    relative_vy = ball2.speedy - ball1.speedy
    # 预测接下来可能的最近距离，通过相对速度和当前距离来计算（简单的线性外推）
    # 这里假设时间步长为1（可根据实际模拟精度需求调整）
    projected_distance = current_distance - (relative_vx * (ball2.pos[0] - ball1.pos[0]) + relative_vy * (ball2.pos[1] - ball1.pos[1])) / current_distance
    return projected_distance <= ball1.radius + ball2.radius


# 计算碰撞后的速度（忽略质量）
def resolve_collision(ball1, ball2):
    ball1.speedx, ball1.speedy, ball2.speedx, ball2.speedy = ball2.speedx,ball2.speedy, ball1.speedx, ball1.speedy
    return ball1,ball2


