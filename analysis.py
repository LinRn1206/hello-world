import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from collections import Counter
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import FIELD_UNITS, OUTPUT_DIR, set_plot_style

def save_plot(filename, dpi=300):
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=dpi)
    plt.close()

def plot_hashtag_distribution(df, topn=20):
    """绘制高频话题标签，并在图上标注数值"""
    hashtags = df['caption'].str.split('#', expand=True).stack().str.strip()
    hashtags = hashtags[hashtags != '']
    top_tags = hashtags.value_counts().head(topn)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=top_tags.values, y=top_tags.index, orient='h')
    ax.set_title(f'Top {topn} 高频话题标签', fontsize=14)
    ax.set_xlabel('出现次数', fontsize=12)
    for i, v in enumerate(top_tags.values):
        ax.text(v, i, f'{int(v):,}',  ha='left',va='center', fontsize=10)
    plt.tight_layout()
    save_plot('top_hashtags.png')

def plot_correlation_heatmap(df):
    """绘制互动指标相关性热力图"""
    corr_matrix = df[list(FIELD_UNITS.keys())[:-1]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                cbar_kws={'label': '相关系数'}, linewidths=0.5)
    plt.title('互动指标相关性分析', fontsize=14)
    plt.xlabel("互动类型")
    plt.ylabel("互动类型")
    save_plot('correlation_heatmap.png')

def plot_activity_patterns(df):
    """组合图：发布时间规律"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    hourly = df['hour'].value_counts().sort_index()
    sns.lineplot(x=hourly.index, y=hourly.values, ax=axes[0], marker='o')
    axes[0].set_title('每日发布时段分布', fontsize=12)
    axes[0].set_xlabel('小时', fontsize=10)
    axes[0].set_ylabel('内容数量（条）', fontsize=10)
    daily = df.resample('D', on='create_time').size()
    sns.lineplot(x=daily.index, y=daily.values, ax=axes[1])
    axes[1].set_title('每日发布量趋势', fontsize=12)
    axes[1].set_xlabel('日期', fontsize=10)
    axes[1].set_ylabel('内容数量（条）', fontsize=10)
    plt.suptitle('用户活跃度分析', fontsize=14)
    save_plot('activity_patterns.png')

def plot_histograms(df):
    """各互动指标分布直方图，单位严格对应"""
    fields = list(FIELD_UNITS.keys())
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(fields, 1):
        plt.subplot(2, 3, i)
        ax = sns.histplot(df[col], bins=50, kde=True)
        ax.set_title(f"{col} 分布")
        ax.set_xlabel(f"{col}{FIELD_UNITS[col]}")
        ax.set_ylabel("内容数量")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.tight_layout()
    save_plot('histograms.png')

def plot_boxplots(df):
    """互动指标箱线图，横轴单位严格对应"""
    fields = list(FIELD_UNITS.keys())
    plt.figure(figsize=(15, 6))
    ax = sns.boxplot(data=df[fields])
    ax.set_title("互动指标箱线图")
    ax.set_xlabel("互动类型")
    ax.set_ylabel("数值/单位")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels([f"{col}{FIELD_UNITS[col]}" for col in fields])
    plt.tight_layout()
    save_plot('boxplots.png')

def plot_top_users(df, topn=10):
    """粉丝数Top用户，横轴单位为人，并在图上标注数值"""
    df['follower_count'] = pd.to_numeric(df['follower_count'], errors='coerce')
    df = df[df['follower_count'] >= 0]
    #df = df[~df['username'].str.contains('官媒', na=False)]
    top_users = df.sort_values('follower_count', ascending=False)
    top_users = top_users.drop_duplicates(subset=['username'])
    top_users = top_users.head(topn)
    # 动态调整高度，每个用户1行，最少6行
    height = max(1.2 * len(top_users), 6)
    plt.figure(figsize=(12, height))
    ax = sns.barplot(x='follower_count', y='username', data=top_users)
    ax.set_title(f"粉丝数Top{topn}用户")
    ax.set_xlabel(f"粉丝数{FIELD_UNITS['follower_count']}")
    ax.set_ylabel("用户名")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    for i, v in enumerate(top_users['follower_count']):
        ax.text(v, i, f'{int(v):,}',  ha='left',va='center', fontsize=10)
    plt.tight_layout()
    save_plot('top_users.png')

def plot_geo_distribution(df):
    """只显示国内IP属地，国外归为'其他'，竖直条形图，顺序随机，纯橙色，'其他'放最后，并标均值红线"""
    china_provinces = [
        "北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
        "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
        "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海",
        "内蒙古", "广西", "西藏", "宁夏", "新疆", "香港", "澳门", "台湾"
    ]
    df['province_clean'] = df['province'].apply(
        lambda x: x if any(p in x for p in china_provinces) else '其他'
    )
    province_counts = df['province_clean'].value_counts()
    if '其他' in province_counts.index:
        other_count = province_counts['其他']
        province_counts = province_counts.drop('其他')
        province_counts = province_counts.sample(frac=1, random_state=42)
        province_counts['其他'] = other_count
    else:
        province_counts = province_counts.sample(frac=1, random_state=42)
    plt.figure(figsize=(max(10, len(province_counts) * 0.5), 8))
    ax = sns.barplot(x=province_counts.index, y=province_counts.values, color='#FFA500')
    ax.set_title("各IP属地内容数量分布（随机顺序，国外归为其他）")
    ax.set_xlabel("IP属地")
    ax.set_ylabel("内容数量（条）")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    for i, v in enumerate(province_counts.values):
        ax.text(i, v, f'{int(v):,}',ha='center', va='bottom', fontsize=9, rotation=90)
    mean_val = province_counts.values.mean()
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label='均值')
    ax.text(len(province_counts)-0.5, mean_val, f'均值：{int(mean_val):,}', color='red', va='bottom', ha='right', fontsize=11)
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    save_plot('geo_distribution.png')

def analyze_and_plot_tags(df, topn=20):
    """分析标签频率与互动指标，并生成相关图表与表格"""
    df['tags'] = df['caption'].str.split('#').apply(
        lambda x: [tag.strip() for tag in x if tag.strip() != '']
    )
    df_exp = df.explode('tags').reset_index(drop=True)
    df_exp = df_exp[df_exp['tags'] != ''][['tags', 'digg_count', 'share_count', 'comment_count']]
    df_exp[['digg_count', 'share_count', 'comment_count']] = df_exp[['digg_count', 'share_count', 'comment_count']].apply(pd.to_numeric, errors='coerce')
    df_exp = df_exp.dropna()
    tag_freq = df_exp['tags'].value_counts().reset_index()
    tag_freq.columns = ['tag', 'frequency']
    tag_stats = df_exp.groupby('tags').agg({
        'digg_count': 'mean',
        'share_count': 'mean',
        'comment_count': 'mean'
    }).reset_index()
    merged = tag_freq.merge(tag_stats, left_on='tag', right_on='tags').drop(columns='tags')
    merged = merged.sort_values(by='frequency', ascending=False)
    merged.to_csv(os.path.join(OUTPUT_DIR, 'tag_analysis.csv'), index=False, encoding='utf-8-sig')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='frequency', y='tag', data=merged.head(topn), palette='viridis', hue='tag', legend=False)
    plt.title(f'Top {topn} 高频标签分布')
    plt.xlabel('出现次数')
    plt.ylabel('标签')
    plt.tight_layout()
    save_plot('tag_frequency.png')
    plt.close()
    plt.figure(figsize=(12, 8))
    heatmap_data = merged.head(topn).set_index('tag')[['digg_count', 'share_count', 'comment_count']]
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlGnBu', linewidths=0.5)
    plt.title('高频标签互动指标热力图（平均值）')
    plt.xlabel('互动类型')
    plt.ylabel('标签')
    plt.tight_layout()
    save_plot('tag_heatmap.png')
    plt.close()
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='frequency',
        y='digg_count',
        size='share_count',
        hue='comment_count',
        data=merged,
        palette='coolwarm',
        sizes=(20, 200),
        alpha=0.7
    )
    plt.title('标签潜力分析：频率 vs 平均点赞数')
    plt.xlabel('标签出现频率')
    plt.ylabel('平均点赞数')
    plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()
    save_plot('tag_potential.png')
    plt.close()

def plot_tag_potential_bubble(merged, topn=50):
    """优化版标签潜力气泡图"""
    data = merged.head(topn)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        data['frequency'], data['digg_count'],
        s=data['comment_count'] * 3 + 20,
        c=data['share_count'], cmap='cool', alpha=0.7, edgecolors='w', linewidths=0.8
    )
    plt.xscale('log')
    plt.xlabel('标签出现频率（对数）')
    plt.ylabel('平均点赞数')
    plt.title('标签潜力气泡图（高点赞/高评论/高分享标签更突出）')
    cbar = plt.colorbar(scatter)
    cbar.set_label('平均分享数')
    top_potential = data.sort_values(['digg_count', 'frequency'], ascending=[False, True]).head(5)
    for _, row in top_potential.iterrows():
        plt.text(row['frequency'], row['digg_count'], row['tag'], fontsize=10, weight='bold')
    plt.tight_layout()
    save_plot('tag_potential_bubble.png')
    plt.close()

def plot_hourly_publish_and_interact(df):
    """24小时发布量与互动量趋势（双Y轴）"""
    hourly = df.groupby('hour').agg({
        'caption': 'count',
        'digg_count': 'mean',
        'comment_count': 'mean'
    }).rename(columns={'caption': '发布量'})
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(hourly.index, hourly['发布量'], 'o-', color='tab:blue', label='发布量')
    ax2.plot(hourly.index, hourly['digg_count'], 's--', color='tab:orange', label='平均点赞')
    ax2.plot(hourly.index, hourly['comment_count'], 'd-.', color='tab:green', label='平均评论')
    ax1.set_xlabel('小时')
    ax1.set_ylabel('发布量', color='tab:blue')
    ax2.set_ylabel('平均互动量', color='tab:orange')
    ax1.set_title('24小时发布量与互动量趋势')
    fig.legend(loc='upper right')
    plt.tight_layout()
    save_plot('hourly_publish_interact.png')
    plt.close()

def plot_monthly_heatmap(df):
    """按月分布的发布量热力图（颜色表示互动强度）"""
    df['year_month'] = df['create_time'].dt.tz_localize(None).dt.to_period('M')
    monthly = df.groupby('year_month').agg({'caption': 'count', 'digg_count': 'mean'})
    plt.figure(figsize=(10, 4))
    sns.heatmap(monthly[['digg_count']].T, annot=True, fmt=".0f", cmap='YlOrRd')
    plt.title('按月分布的发布量热力图（互动强度）')
    plt.xlabel('月份')
    plt.ylabel('互动类型')
    plt.tight_layout()
    save_plot('monthly_heatmap.png')
    plt.close()

def time_interaction_correlation(df):
    """发布时间与互动量相关性分析"""
    corr = df[['hour', 'digg_count', 'comment_count', 'share_count']].corr()
    print("发布时间与互动量相关性：\n", corr['hour'])
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='hour', y='digg_count', data=df, alpha=0.3)
    plt.title('发布时间与点赞数散点图')
    plt.tight_layout()
    save_plot('hour_vs_digg.png')
    plt.close()

def analyze_follower_interaction(df):
    """粉丝规模与互动量关系分析及小而美账号识别"""
    numeric_cols = ['follower_count', 'digg_count', 'share_count', 'comment_count']
    if not all(col in df.columns for col in numeric_cols + ['username']):
        print("缺少必要字段，无法分析粉丝互动关系")
        return
    df_clean = df[numeric_cols + ['username']].copy()
    df_clean[numeric_cols] = df_clean[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_clean = df_clean.dropna()
    df_clean = df_clean[df_clean['follower_count'] > 0]
    user_stats = df_clean.groupby('username').agg({
        'follower_count': 'max',
        'digg_count': 'mean',
        'share_count': 'mean',
        'comment_count': 'mean'
    }).reset_index()
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=user_stats,
        x='follower_count',
        y='digg_count',
        hue='share_count',
        size='comment_count',
        sizes=(20, 200),
        alpha=0.6,
        palette='viridis'
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.title('粉丝规模与互动量关系（对数坐标）')
    plt.xlabel('粉丝数（对数）')
    plt.ylabel('平均点赞数（对数）')
    plt.grid(True)
    save_plot('follower_interaction_scatter.png')
    plt.close()
    correlation = user_stats[['follower_count', 'digg_count', 'share_count', 'comment_count']].corr(method='pearson')
    print("粉丝数与互动指标的相关系数：\n", correlation['follower_count'])
    median_digg = user_stats['digg_count'].median()
    small_but_high = user_stats[
        (user_stats['follower_count'] < 10000) &
        (user_stats['digg_count'] > median_digg)
    ]
    small_but_high.to_csv(os.path.join(OUTPUT_DIR, 'small_but_high_impact.csv'), index=False, encoding='utf-8-sig')
    print(f"已识别小而美账号数量：{len(small_but_high)}，结果已保存至 {os.path.join(OUTPUT_DIR, 'small_but_high_impact.csv')}")

def analyze_geo_distribution_and_preference(df, wordcloud_province='江西'):
    """地理分布、内容偏好与互动对比分析"""
    df['province'] = df['ip_location'].str.replace('IP属地：', '', regex=False)
    df = df[df['province'] != '']
    province_activity = df['province'].value_counts().reset_index()
    province_activity.columns = ['province', 'post_count']
    top_provinces = province_activity.head(10)
    df['tags'] = df['caption'].str.split('#').apply(
        lambda x: [tag.strip() for tag in x if tag.strip()]
    )
    df_exp = df.explode('tags')
    def get_top_tags(group):
        return Counter(group['tags']).most_common(3)
    province_tags = df_exp.groupby('province').apply(get_top_tags).reset_index()
    province_tags.columns = ['province', 'top_tags']
    plt.figure(figsize=(12, 6))
    sns.barplot(x='post_count', y='province', data=top_provinces, hue='province', palette='Blues_d', legend=False)
    plt.title('Top 10 地区内容发布量')
    plt.xlabel('内容数量')
    plt.ylabel('地区')
    plt.tight_layout()
    save_plot('geo_activity.png')
    plt.close()
    province_tags_list = df_exp[df_exp['province'] == wordcloud_province]['tags']
    if not province_tags_list.empty:
        tag_freq = Counter(province_tags_list)
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='msyh.ttc').generate_from_frequencies(tag_freq)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{wordcloud_province}省高频内容标签词云')
        save_plot(f'{wordcloud_province}_tags.png')
        plt.close()
    all_tags = df_exp['tags'].dropna()
    if not all_tags.empty:
        all_tag_freq = Counter(all_tags)
        wordcloud = WordCloud(width=1000, height=500, background_color='white', font_path='msyh.ttc').generate_from_frequencies(all_tag_freq)
        plt.figure(figsize=(14, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('全局内容标签词云')
        save_plot('all_tags_wordcloud.png')
        plt.close()
    province_interaction = df.groupby('province').agg({
        'digg_count': 'mean',
        'share_count': 'mean',
        'comment_count': 'mean'
    }).reset_index()
    top_provinces_list = top_provinces['province'].tolist()
    province_interaction_top = province_interaction[province_interaction['province'].isin(top_provinces_list)]
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        province_interaction_top.set_index('province'),
        annot=True, fmt=".1f", cmap='YlGnBu', linewidths=0.5
    )
    plt.title('高活跃地区平均互动量对比')
    plt.xlabel('互动类型')
    plt.ylabel('地区')
    plt.tight_layout()
    save_plot('geo_interaction.png')
    plt.close()

def analyze_interaction_correlation(df):
    """
    分析互动指标（点赞、收藏、评论、分享）与视频时长（time）的相关性，
    并可视化相关系数矩阵与关键散点图。
    """
    cols = ['digg_count', 'collect_count', 'comment_count', 'share_count', 'time']
    for col in cols:
        if col not in df.columns:
            print(f"缺少字段: {col}，无法进行相关性分析")
            return
    df_corr = df[cols].apply(pd.to_numeric, errors='coerce')
    df_corr = df_corr.dropna()
    df_corr = df_corr[(df_corr >= 0).all(axis=1)]
    for col in cols:
        upper = df_corr[col].quantile(0.99)
        df_corr = df_corr[df_corr[col] <= upper]
    pearson_corr = df_corr.corr(method='pearson')
    spearman_corr = df_corr.corr(method='spearman')
    print("皮尔逊相关系数：\n", pearson_corr)
    print("斯皮尔曼秩相关系数：\n", spearman_corr)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('互动指标与视频时长相关性热力图（皮尔逊）')
    save_plot('interaction_time_corr_heatmap.png')
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x='digg_count', y='collect_count', data=df_corr, alpha=0.3)
    plt.title('点赞数与收藏数关系')
    plt.xlabel('点赞数')
    plt.ylabel('收藏数')
    plt.tight_layout()
    save_plot('digg_vs_collect_scatter.png')
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x='time', y='collect_count', data=df_corr, alpha=0.3)
    sns.regplot(x='time', y='collect_count', data=df_corr, scatter=False, color='red', lowess=True)
    plt.title('视频时长与收藏数关系')
    plt.xlabel('视频时长（秒）')
    plt.ylabel('收藏数')
    plt.tight_layout()
    save_plot('time_vs_collect_scatter.png')
    print("\n【相关性分析结果解读建议】")
    print("- 点赞与收藏的相关系数（皮尔逊）为：", round(pearson_corr.loc['digg_count', 'collect_count'], 3))
    print("- 视频时长与收藏的相关系数（皮尔逊）为：", round(pearson_corr.loc['time', 'collect_count'], 3))
    print("  若为正且较大，说明长视频更易被收藏；若为负或接近0，则无明显偏好。")
    print("- 建议结合散点图和回归线观察是否存在非线性关系。")
    print("- 若数据分布偏态严重，可优先参考斯皮尔曼秩相关系数。")

def cluster_content(df, n_clusters=4):
    """内容聚类分析：基于互动、标签主题、用户属性"""
    num_cols = ['digg_count', 'collect_count', 'comment_count', 'share_count', 'follower_count']
    X_num = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    if 'caption_topic' in df.columns:
        topic_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_topic = topic_ohe.fit_transform(df[['caption_topic']].fillna(-1))
    else:
        X_topic = np.zeros((len(df), 1))
    if 'tags' in df.columns:
        df['main_tag'] = df['tags'].apply(lambda x: x[0] if isinstance(x, list) and x else '无')
        tag_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_tag = tag_ohe.fit_transform(df[['main_tag']])
    else:
        X_tag = np.zeros((len(df), 1))
    X = np.hstack([X_num_scaled, X_topic, X_tag])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    df['cluster'] = cluster_labels
    score = silhouette_score(X, cluster_labels)
    print(f"K-means聚类轮廓系数：{score:.3f}")
    cluster_summary = df.groupby('cluster')[num_cols + ['caption_sentiment']].mean()
    print("各簇互动与情感均值：\n", cluster_summary)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_summary, annot=True, fmt=".1f", cmap='YlGnBu')
    plt.title('不同内容簇的互动与情感均值')
    plt.tight_layout()
    save_plot('cluster_feature_heatmap.png')
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='digg_count', y='comment_count', hue='cluster', data=df, palette='tab10', alpha=0.6)
    plt.title('内容聚类分布（点赞-评论）')
    plt.tight_layout()
    save_plot('cluster_scatter.png')
    if 'main_tag' in df.columns:
        tag_dist = df.groupby('cluster')['main_tag'].agg(lambda x: x.value_counts().index[0])
        print("各簇主标签：\n", tag_dist)
    print("\n【内容聚类分析建议】")
    print("- 可根据各簇的互动均值、主标签、情感倾向，命名如“高互动搞笑类”“低互动科普类”等。")
    print("- 针对高互动簇加大内容投放，低互动簇可优化内容形式或发布时间。")