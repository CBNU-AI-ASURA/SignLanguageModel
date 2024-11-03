import os
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정 (예: 맑은 고딕)
font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'  # Mac OS 예시
# Windows 사용자라면 'C:/Windows/Fonts/malgun.ttf' 등으로 변경
if os.path.exists(font_path):
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
else:
    print("한글 폰트 파일을 찾을 수 없습니다. 시각화에서 한글이 제대로 표시되지 않을 수 있습니다.")