# 实验绘图系统

这是一个基于Hydra的可扩展实验绘图系统，用于绘制和比较实验结果。它提供了一个高度可配置和可扩展的方式来生成不同类型的图表，而无需重复编写代码。

## 特性

- 使用Hydra进行配置管理
- 支持多种预定义的绘图类型
- 易于扩展的绘图注册系统
- 灵活的数据加载和处理
- 可自定义的样式和输出格式

## 安装和依赖

该系统依赖于以下Python包：
- hydra-core >= 1.3.2
- matplotlib
- numpy
- omegaconf

## 使用方法

### 基本用法

#### 使用Hydra配置方式

```bash
# 使用默认配置
python -m plotter.main

# 指定绘图类型
python -m plotter.main plot=experiment_comparison

# 覆盖配置中的特定值
python -m plotter.main plot=rollout_evaluation plot.experiment_dir=path/to/experiment

# 使用多组实验目录
python -m plotter.main plot=experiment_comparison "plot.experiment_dirs=[path1,path2,path3]"

# 更改输出格式
python -m plotter.main plot=experiment_comparison plot.render_mode=svg
```

#### 使用简化的命令行工具

我们还提供了一个更简单的命令行工具，无需记忆Hydra的语法：

```bash
# 比较多个实验
python -m plotter.plot compare --dirs path1 path2 path3 --title "My Comparison" --output png

# 查看rollout和evaluation对比
python -m plotter.plot rollout --dir path/to/experiment --title "Rollout vs Eval"

# 绘制学习曲线（带移动平均）
python -m plotter.plot learning --dirs path1 path2 --window 15 --ylim 0 5000

# 显示帮助
python -m plotter.plot --help
```

### 可用的绘图类型

1. **experiment_comparison**: 比较多个实验的结果（命令行别名：`compare`, `exp`）
2. **rollout_evaluation**: 对比rollout和evaluation性能（命令行别名：`rollout`, `eval`）
3. **exception_rate**: 分析异常率（命令行别名：`exception`）
4. **custom**: 完全自定义的绘图配置
5. **learning_curve**: 绘制带有移动平均的学习曲线（命令行别名：`learning`, `curve`）
6. **ablation_study**: 消融研究的柱状图比较（命令行别名：`ablation`）

## 特殊路径处理

该绘图系统支持使用通配符（`*`）来匹配多个实验目录，这在处理具有随机种子的实验目录时特别有用。例如：

```bash
# 使用Hydra方式
python -m plotter.main plot=experiment_comparison "plot.experiment_dirs=[/path/to/exp*seed*]"

# 使用命令行工具
python -m plotter.plot compare --dirs "/path/to/exp*seed*" "/path/to/another*"
```

系统会自动查找所有匹配的目录，并将它们视为同一实验的不同运行，计算平均值和范围。

## 配置系统

### 配置文件层次结构

```
plotter/
└── conf/
    ├── config.yaml             # 主配置文件
    ├── plot/                   # 绘图类型配置
    │   ├── experiment_comparison.yaml
    │   ├── rollout_evaluation.yaml
    │   ├── exception_rate.yaml
    │   ├── custom.yaml
    │   ├── learning_curve.yaml
    │   └── ablation_study.yaml
    └── data/                   # 数据源配置
```

### 添加新的绘图类型

1. 创建一个新的绘图函数并使用装饰器注册它：

```python
from plotter.plot_manager import PlotRegistry, DictConfig

@PlotRegistry.register('my_new_plot_type')
def plot_my_new_type(cfg: DictConfig) -> None:
    # 实现你的绘图逻辑
    ...
```

2. 在`conf/plot/`目录下创建一个新的配置文件：

```yaml
# @package _global_.plot

# 绘图类型
plot_type: my_new_plot_type

# 你的配置参数
...
```

3. 现在可以使用你的新绘图类型：

```bash
python -m plotter.main plot=my_new_plot_type
```

## 扩展示例

查看`plotter/examples/extend_plots.py`文件，了解如何扩展绘图系统并添加新的绘图类型。

## 最佳实践

1. **使用配置文件而非硬编码参数**：将所有可变参数放在配置文件中，使代码更具可维护性。

2. **创建可复用的绘图组件**：将常用的绘图功能封装为函数，以便在不同的绘图类型中重用。

3. **使用有意义的命名**：为实验目录、图表和参数使用描述性名称，使配置更容易理解。

4. **版本控制你的配置**：将重要的配置文件保存在版本控制系统中，以便跟踪实验设置的变化。

5. **利用Hydra的组合功能**：使用Hydra的组合功能创建不同的配置组合，而不是复制配置文件。

6. **使用命令行工具进行快速绘图**：对于日常使用，推荐使用`plotter.plot`命令行工具，它提供了更简洁的接口。

7. **对多次实验运行使用通配符**：使用通配符路径（如`/path/to/exp*seed-*`）来一次性处理多个实验运行。