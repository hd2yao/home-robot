import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import platform
import os

def configure_chinese_font():
    """配置matplotlib使用支持中文的字体"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        # 常见的macOS中文字体
        chinese_fonts = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Microsoft/SimHei.ttf',
            '/Library/Fonts/Songti.ttc'
        ]
    elif system == 'Windows':
        # 常见的Windows中文字体
        chinese_fonts = [
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/simsun.ttc',
            'C:/Windows/Fonts/simkai.ttf',
            'C:/Windows/Fonts/msyh.ttc'
        ]
    elif system == 'Linux':
        # 常见的Linux中文字体
        chinese_fonts = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        ]
    else:
        chinese_fonts = []
    
    # 查找第一个存在的中文字体
    font_path = None
    for font in chinese_fonts:
        if os.path.exists(font):
            font_path = font
            break
    
    if font_path:
        # 设置matplotlib使用找到的中文字体
        font_properties = FontProperties(fname=font_path)
        matplotlib.rcParams['font.family'] = font_properties.get_name()
        print(f"已设置matplotlib使用字体: {font_path}")
        return True
    else:
        # 如果找不到中文字体，使用matplotlib的内置字体
        print("未找到支持中文的字体，将尝试使用matplotlib内置字体")
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return False

if __name__ == "__main__":
    configure_chinese_font()
    # 测试中文显示
    plt.figure(figsize=(8, 6))
    plt.title('中文显示测试')
    plt.xlabel('横坐标')
    plt.ylabel('纵坐标')
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.savefig('chinese_font_test.png')
    plt.close()
    print("测试图片已保存为 chinese_font_test.png") 