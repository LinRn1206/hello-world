import matplotlib

# 字段与单位映射
FIELD_UNITS = {
    'collect_count': '（次）',
    'comment_count': '（条）',
    'digg_count': '（次）',
    'share_count': '（次）',
    'follower_count': '（人）'
}

# 停用词
STOPWORDS = set(['的', '了', '和', '是', '在', '我', '有', '也'])

# 输出路径
OUTPUT_DIR = 'output'

# 全局样式设置
def set_plot_style():
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False