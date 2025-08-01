"""
可扩展的实验绘图系统。

该包提供了一个基于Hydra的配置系统，用于生成各种实验结果的图表。
主要组件包括：
- 数据加载器：从各种文件格式加载数据
- 绘图管理器：基于配置执行绘图过程
- 绘图类型注册表：提供可扩展的绘图类型系统

使用方法:
    python -m plotter.main plot=experiment_comparison
"""

from plotter.plot_manager import PlotManager, PlotRegistry, DataLoader

__all__ = ["PlotManager", "PlotRegistry", "DataLoader"]