# 迁移指南：从旧绘图代码到新绘图系统

本指南将帮助您将旧的绘图代码迁移到新的基于Hydra的可扩展绘图系统。

## 概述

新的绘图系统提供了以下优势：

1. **配置驱动**：使用Hydra管理配置，无需修改代码即可调整绘图参数
2. **可扩展性**：简单注册新的绘图类型，支持快速添加新的实验图表
3. **代码重用**：消除冗余代码，提高可维护性
4. **一致的样式**：集中管理样式配置，确保图表一致性

## 迁移步骤

### 步骤 1: 识别现有的绘图模式

从旧代码中，我们可以观察到以下常见模式：

- 从CSV文件读取数据
- 多个实验结果的比较
- Rollout和Evaluation的对比
- 绘制基准线和特殊点
- 计算平均值和范围

### 步骤 2: 选择或创建合适的配置文件

新系统已经包含多种预定义的绘图类型：

| 旧函数 | 新配置文件 | 描述 |
|-------|-----------|------|
| `plot_experiment` | `experiment_comparison.yaml` | 比较多个实验结果 |
| `plot_rollout_evaluation` | `rollout_evaluation.yaml` | 比较rollout和evaluation性能 |
| `plot_exception_rate` | `exception_rate.yaml` | 分析异常率 |
| 其他自定义图表 | `custom.yaml` | 完全自定义的绘图 |

### 步骤 3: 转换具体实例

以下是一些常见的转换示例：

#### 示例 1: `plot_experiment` 转换

旧代码:
```python
plot_experiment(
    online_experiment_dirs=["exp1", "exp2", "exp3"],
    plot_name="My Experiment",
    colors=["blue", "red", "green"],
    labels=["Algo1", "Algo2", "Algo3"],
    render_mode="svg",
    save_path="./plots"
)
```

新系统 (命令行):
```bash
python -m plotter.main plot=experiment_comparison \
    "plot.experiment_dirs=[exp1,exp2,exp3]" \
    plot.title="My Experiment" \
    "plot.colors=[blue,red,green]" \
    "plot.labels=[Algo1,Algo2,Algo3]" \
    plot.render_mode=svg \
    plot.save_path=./plots
```

或者创建一个新的配置文件 `my_experiment.yaml`:
```yaml
# @package _global_.plot
plot_type: experiment_comparison
experiment_dirs:
  - exp1
  - exp2
  - exp3
title: "My Experiment"
colors:
  - blue
  - red
  - green
labels:
  - "Algo1"
  - "Algo2"
  - "Algo3"
render_mode: svg
save_path: ./plots
```

然后运行:
```bash
python -m plotter.main plot=my_experiment
```

#### 示例 2: `plot_rollout_evaluation` 转换

旧代码:
```python
plot_rollout_evaluation(
    experiment_dir="my_experiment",
    plot_name="Rollout vs Evaluation"
)
```

新系统:
```bash
python -m plotter.main plot=rollout_evaluation \
    plot.experiment_dir=my_experiment \
    plot.title="Rollout vs Evaluation"
```

### 步骤 4: 处理自定义绘图函数

对于更特殊的绘图函数，您可以：

1. **使用已有的 `custom` 配置类型**：适用于简单的自定义绘图
2. **注册新的绘图类型**：对于复杂的绘图逻辑

示例: 注册新的绘图类型
```python
# 在plotter/examples/extend_plots.py中添加
from plotter.plot_manager import PlotRegistry, DictConfig

@PlotRegistry.register('my_special_plot')
def plot_my_special_type(cfg: DictConfig) -> None:
    # 实现您的绘图逻辑，类似于旧函数
    experiment_dirs = cfg.experiment_dirs
    # ...其余代码...
```

然后创建配置文件 `conf/plot/my_special_plot.yaml`

## 常见问题

### Q: 如何添加新的数据源类型？
在 `plotter/plot_manager.py` 中的 `DataLoader` 类中添加新的加载方法。

### Q: 如何保持特定的格式或样式？
在配置文件中设置样式参数，或修改 `PlotStyle` 类添加新的样式支持。

### Q: 如何批量生成多个图表？
使用Hydra的多运行功能:
```bash
python -m plotter.main --multirun plot=experiment_comparison,rollout_evaluation
```

## 完整示例

查看 `plotter/examples/demo.py` 获取完整的示例代码，包括如何在代码中使用新的绘图系统。

## 注意事项

1. **路径处理**：确保实验目录路径正确，相对路径或绝对路径都可以使用
2. **默认值**：新系统为大多数参数提供了合理的默认值
3. **扩展性**：如有新的绘图需求，请注册新的绘图类型而非修改现有类型

---

如需更多帮助，请参考 `plotter/README.md` 或示例文件。