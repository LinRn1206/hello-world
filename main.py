import os
import json
import pandas as pd
from config import set_plot_style, OUTPUT_DIR
from preprocess import preprocess_data, preprocess_time_fields, enrich_text_features
from analysis import (
    plot_hashtag_distribution, plot_correlation_heatmap, plot_activity_patterns,
    plot_histograms, plot_boxplots, plot_top_users, plot_geo_distribution,
    analyze_and_plot_tags, plot_hourly_publish_and_interact, plot_monthly_heatmap,
    analyze_follower_interaction, cluster_content, analyze_geo_distribution_and_preference,
    analyze_interaction_correlation
)

def main():
    set_plot_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open('processed.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data)
    df = preprocess_data(df)
    df = preprocess_time_fields(df)
    df = enrich_text_features(df)

    analysis_functions = [
        plot_hashtag_distribution,
        plot_correlation_heatmap,
        analyze_interaction_correlation,
        plot_activity_patterns,
        plot_histograms,
        plot_boxplots,
        plot_top_users,
        plot_geo_distribution,
        analyze_and_plot_tags,
        plot_hourly_publish_and_interact,
        plot_monthly_heatmap,
        analyze_follower_interaction,
        cluster_content,
    ]
    for func in analysis_functions:
        try:
            func(df)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
    print("分析完成，图表已保存至 output 目录")
    try:
        analyze_geo_distribution_and_preference(df, wordcloud_province='湖南')
    except Exception as e:
        print(f"Error in analyze_geo_distribution_and_preference: {str(e)}")

if __name__ == "__main__":
    main()