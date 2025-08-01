#!/usr/bin/env python3
"""
扩展绘图系统示例

本示例展示如何注册新的绘图类型并创建相应的配置文件
"""
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plotter.plot_manager import PlotRegistry, DataLoader, PlotTypes
from omegaconf import DictConfig


# 注册一个新的绘图类型
@PlotRegistry.register('learning_curve')
def plot_learning_curve(cfg: DictConfig) -> None:
    """
    绘制学习曲线，带有移动平均线
    
    配置参数:
    - experiment_dirs: 实验目录列表
    - colors: 颜色列表
    - labels: 标签列表
    - filename: 数据文件名
    - window_size: 移动平均窗口大小
    """
    experiment_dirs = cfg.experiment_dirs
    colors = cfg.get('colors', ['blue'] * len(experiment_dirs))
    labels = cfg.get('labels', [f"Experiment {i+1}" for i in range(len(experiment_dirs))])
    filename = cfg.get('filename', 'evaluation.csv')
    window_size = cfg.get('window_size', 10)
    
    for exp_dir, color, label in zip(experiment_dirs, colors, labels):
        file_paths = [f"{exp_dir}/{filename}"]
        all_y = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                x, y = DataLoader.read_csv(file_path)
                all_y.append(y)
        
        if all_y:
            # 原始数据
            raw_data = np.mean(all_y, axis=0)
            
            # 计算移动平均
            smoothed_data = []
            for i in range(len(raw_data)):
                start = max(0, i - window_size + 1)
                smoothed_data.append(np.mean(raw_data[start:i+1]))
            
            # 绘制原始数据（淡色）
            PlotTypes.line_plot(x, raw_data, DictConfig(dict(
                color=color,
                alpha=0.3,
                linewidth=0.5,
                label=f"{label} (Raw)"
            )))
            
            # 绘制平滑后的数据
            PlotTypes.line_plot(x, smoothed_data, DictConfig(dict(
                color=color,
                linewidth=2.0,
                label=f"{label} (Smoothed)"
            )))


# 注册另一个自定义绘图类型
@PlotRegistry.register('ablation_study')
def plot_ablation_study(cfg: DictConfig) -> None:
    """
    绘制消融研究的柱状图
    
    配置参数:
    - experiment_names: 实验名称列表
    - values: 各实验的性能值
    - error_bars: 误差条（可选）
    - colors: 颜色列表
    """
    import matplotlib.pyplot as plt
    
    experiment_names = cfg.experiment_names
    values = cfg.values
    error_bars = cfg.get('error_bars', [0] * len(values))
    colors = cfg.get('colors', ['blue'] * len(values))
    
    # 创建柱状图
    plt.bar(
        range(len(experiment_names)), 
        values, 
        yerr=error_bars,
        color=colors,
        capsize=10
    )
    
    # 设置x轴标签
    plt.xticks(range(len(experiment_names)), experiment_names, rotation=45)
    
    # 在柱子上方标注数值
    for i, v in enumerate(values):
        plt.text(i, v + max(error_bars[i], 5), f"{v:.1f}", 
                 ha='center', va='bottom', fontweight='bold')


if __name__ == "__main__":
    print("此文件展示了如何扩展绘图系统")
    print("您可以通过以下命令使用新添加的绘图类型：")
    print("python -m plotter.main plot=learning_curve")
    print("python -m plotter.main plot=ablation_study")
    print("请确保在conf/plot/目录下创建了相应的配置文件")