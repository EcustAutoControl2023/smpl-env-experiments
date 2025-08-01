# 实验绘图配置

本文档介绍了从`plot_experiments.py`中提取出来的各种实验绘图配置，使用YAML格式方便查看和修改。

## 使用方法

### 通过Hydra命令行运行

```bash
# 使用基本的实验比较配置
python -m plotter.main plot=exp_com_expert

# 比较不同比例的专家指导实验
python -m plotter.main plot=exp_com_expert

# 绘制离线+在线实验的对比
python -m plotter.main plot=exp_com_offline_online

# 修改特定配置参数
python -m plotter.main plot=exp_com_expert "plot.title=我的实验对比" "plot.ylim=[-100,6000]"
```

### 通过简化的命令行工具运行

```bash
# 比较不同比例的专家指导实验
python -m plotter.plot custom --config exp_com_expert

# 绘制离线+在线实验的对比
python -m plotter.plot custom --config exp_com_offline_online
```

## 配置文件列表

### 专家指导和在线/离线实验对比

- `exp_com_expert.yaml`: 不同比例专家指导的对比
- `exp_com_expert_online.yaml`: 仅在线学习中的专家指导对比
- `exp_com_offline_online.yaml`: 在线学习vs离线+在线学习对比

### 不同比例的专家指导实验

- `exp_com_online_offline_rate01.yaml`: 10%专家指导率的对比
- `exp_com_online_offline_rate03.yaml`: 30%专家指导率的对比
- `exp_com_online_offline_rate05.yaml`: 50%专家指导率的对比
- `exp_com_online_offline_rate07.yaml`: 70%专家指导率的对比

### 离线/在线方法组合实验

- `exp_diff_offline_guiding_online.yaml`: 不同离线算法引导在线学习的对比
- `exp_offline_improvement.yaml`: 错误代码处理的对比
- `exp_experiment_oo.yaml`: 离线+在线实验比较

### 其他环境实验

- `exp_experiment_reactor.yaml`: Reactor环境实验比较
- `exp_reactor.yaml`: Reactor环境的实验
- `exp_mabenv_plot.yaml`: MAB环境的实验

## 自定义和扩展

你可以根据需要修改这些配置文件，或者基于它们创建新的配置。配置文件使用YAML格式，结构清晰易懂。

主要配置项包括：
- `plot_type`: 绘图类型
- `experiment_dirs`: 实验目录列表
- `title`, `xlabel`, `ylabel`: 图表标题和轴标签
- `ylim`: Y轴范围
- `colors`, `labels`: 线条颜色和标签
- `baseline_value`: 基准线值
- `offline_value`: 离线值点
- `render_mode`: 渲染模式（plot, svg, png, pdf）
- `save_path`: 保存路径

## 原始代码映射

这些配置文件是从`plot_experiments.py`中的以下函数转换而来：

- `plot_experiment` → `experiment_comparison.yaml`
- `plot_experiment_reactor` → `experiment_reactor.yaml` 
- `plot_experiment_oo` → `experiment_oo.yaml`
- `plot_com_expert` → `exp_com_expert.yaml`
- `plot_com_expert_online` → `exp_com_expert_online.yaml`
- `plot_com_offline_online` → `exp_com_offline_online.yaml`
- `plot_reactor` → `exp_reactor.yaml`
- `plot_diff_offline_guiding_online` → `exp_diff_offline_guiding_online.yaml`
- `plot_offline_imrovement` → `exp_offline_improvement.yaml`
- `mabenv_plot` → `exp_mabenv_plot.yaml`

各种比例的专家指导实验是从`plot_com_online_offline`函数转换而来。