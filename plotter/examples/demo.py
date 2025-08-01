#!/usr/bin/env python3
"""
绘图系统使用示例

本脚本展示如何使用重构后的绘图系统来生成各种类型的图表。
"""
import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from omegaconf import DictConfig
from plotter.plot_manager import PlotManager, DataLoader


def generate_sample_data():
    """生成示例数据文件用于演示"""
    # 创建临时目录
    base_dir = tempfile.mkdtemp(prefix="plot_demo_")
    
    # 创建几个实验目录
    for exp_id in range(1, 4):
        exp_dir = os.path.join(base_dir, f"experiment{exp_id}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # 创建评估数据
        x = list(range(0, 100, 1))
        
        # 不同的实验有不同的性能曲线
        if exp_id == 1:
            # 快速收敛但较低的性能
            y = [3000 * (1 - np.exp(-0.1 * i)) + np.random.normal(0, 100) for i in x]
        elif exp_id == 2:
            # 慢速收敛但较高的性能
            y = [4000 * (1 - np.exp(-0.05 * i)) + np.random.normal(0, 150) for i in x]
        else:
            # 中等收敛，中等性能
            y = [3500 * (1 - np.exp(-0.07 * i)) + np.random.normal(0, 125) for i in x]
        
        # 创建评估CSV文件
        with open(os.path.join(exp_dir, "evaluation.csv"), "w") as f:
            for i, val in zip(x, y):
                f.write(f"{i},0,{val}\n")
        
        # 创建rollout CSV文件 (通常低于评估值)
        with open(os.path.join(exp_dir, "rollout_return.csv"), "w") as f:
            for i, val in zip(x, y):
                rollout_val = val * 0.8 + np.random.normal(0, 200)  # 更嘈杂
                f.write(f"{i},0,{rollout_val}\n")
    
    return base_dir


def demo_experiment_comparison(base_dir):
    """演示实验比较图"""
    print("演示: 实验比较图")
    
    # 创建配置
    cfg = DictConfig({
        "plot_type": "experiment_comparison",
        "experiment_dirs": [
            f"{base_dir}/experiment1",
            f"{base_dir}/experiment2",
            f"{base_dir}/experiment3"
        ],
        "filename": "evaluation.csv",
        "title": "Demo: Experiment Comparison",
        "xlabel": "Epoch",
        "ylabel": "Evaluation Return",
        "ylim": [0, 5000],
        "colors": ["blue", "red", "green"],
        "labels": ["Fast Convergence", "Slow Convergence", "Medium Convergence"],
        "baseline_value": 3500,
        "offline_value": 2000,
        "render_mode": "plot",
        "show_range": True,
        "legend": True
    })
    
    # 创建并执行绘图管理器
    plot_manager = PlotManager(cfg)
    plot_manager.execute()


def demo_rollout_evaluation(base_dir):
    """演示rollout与评估对比图"""
    print("演示: Rollout vs Evaluation 对比图")
    
    # 创建配置
    cfg = DictConfig({
        "plot_type": "rollout_evaluation",
        "experiment_dir": f"{base_dir}/experiment2",
        "filenames": ["rollout_return.csv", "evaluation.csv"],
        "title": "Demo: Rollout vs Evaluation",
        "xlabel": "Epoch",
        "ylabel": "Return",
        "ylim": [0, 5000],
        "colors": ["blue", "red"],
        "labels": ["Rollout", "Evaluation"],
        "render_mode": "plot",
        "show_range": True,
        "legend": True
    })
    
    # 创建并执行绘图管理器
    plot_manager = PlotManager(cfg)
    plot_manager.execute()


def demo_custom_plot():
    """演示自定义绘图"""
    print("演示: 自定义绘图")
    
    # 创建配置
    cfg = DictConfig({
        "plot_type": "custom",
        "title": "Demo: Custom Plot",
        "xlabel": "Steps",
        "ylabel": "Performance",
        "ylim": [0, 100],
        "xlim": [0, 100],
        "instructions": [
            {
                "function": "line_plot",
                "params": {
                    "x": list(range(0, 101, 10)),
                    "y": [0, 20, 35, 48, 58, 65, 72, 78, 83, 87, 90],
                    "color": "blue",
                    "label": "Algorithm A"
                }
            },
            {
                "function": "line_plot",
                "params": {
                    "x": list(range(0, 101, 10)),
                    "y": [0, 10, 22, 36, 52, 65, 75, 84, 90, 94, 97],
                    "color": "red",
                    "label": "Algorithm B",
                    "linestyle": "--"
                }
            },
            {
                "function": "scatter_plot",
                "params": {
                    "x": [50, 80],
                    "y": [65, 84],
                    "color": "green",
                    "label": "Key Points",
                    "marker": "*",
                    "size": 200
                }
            },
            {
                "function": "horizontal_line",
                "params": {
                    "y_value": 75,
                    "color": "black",
                    "linestyle": ":",
                    "label": "Target Performance"
                }
            }
        ],
        "render_mode": "plot",
        "legend": True
    })
    
    # 创建并执行绘图管理器
    plot_manager = PlotManager(cfg)
    plot_manager.execute()


def demo_learning_curve(base_dir):
    """演示学习曲线绘图"""
    print("演示: 学习曲线（带移动平均）")
    
    # 创建配置
    cfg = DictConfig({
        "plot_type": "learning_curve",
        "experiment_dirs": [
            f"{base_dir}/experiment1",
            f"{base_dir}/experiment3"
        ],
        "filename": "evaluation.csv",
        "window_size": 10,
        "title": "Demo: Learning Curves with Moving Average",
        "xlabel": "Epoch",
        "ylabel": "Evaluation Return",
        "ylim": [0, 5000],
        "colors": ["blue", "green"],
        "labels": ["Fast Convergence", "Medium Convergence"],
        "render_mode": "plot",
        "legend": True
    })
    
    # 创建并执行绘图管理器
    plot_manager = PlotManager(cfg)
    plot_manager.execute()


def main():
    """主函数，运行所有演示"""
    print("绘图系统演示")
    print("============")
    
    # 生成示例数据
    base_dir = generate_sample_data()
    print(f"生成的示例数据位于: {base_dir}")
    
    try:
        # 运行各种演示
        demo_experiment_comparison(base_dir)
        demo_rollout_evaluation(base_dir)
        demo_custom_plot()
        demo_learning_curve(base_dir)
        
        print("\n所有演示完成!")
        print(f"您可以在 {base_dir} 目录下查看生成的示例数据")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
    
    # 注意：此处不删除临时目录，以便查看生成的数据


if __name__ == "__main__":
    main()