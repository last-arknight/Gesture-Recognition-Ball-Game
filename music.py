import pygame
import random

# 初始化pygame的音频模块
class Music():
    def __init__(self):
        pygame.mixer.init()
    # 音乐文件路径
        # self.music_files = ["musics/1.mp3", "musics/2.mp3", "musics/3.mp3","musics/4.mp3","musics/5.mp3"]  # 替换为你自己的音乐文件路径
        self.music_files = ["musics/1.mp3"]  # 替换为你自己的音乐文件路径
        self.current_music = None
        pygame.mixer.music.stop()

    # 播放/暂停音乐的功能
    def toggle_music(self):
        if self.current_music is None:
            # 随机播放一首音乐
            self.current_music = random.choice(self.music_files)
            pygame.mixer.music.load(self.current_music)
            pygame.mixer.music.play(-1)  # 循环播放
        else:
            # 关闭音乐
            pygame.mixer.music.stop()
            current_music = None

    # 切换音乐的功能
    def switch_music(self):
        # 确保当前播放的音乐不会被重新选择
        if self.current_music:
            remaining_music = [music for music in self.music_files if music != self.current_music]
        else:
            remaining_music = self.music_files

        # 如果剩余的音乐不为空，从中随机选择一首
        if remaining_music:
            self.current_music = random.choice(remaining_music)
            pygame.mixer.music.load(self.current_music)  # 加载新音乐
            pygame.mixer.music.play(-1)  # 循环播放
