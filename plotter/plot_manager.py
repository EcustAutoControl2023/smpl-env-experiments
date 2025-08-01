import csv
import glob
import os
import logging
from typing import Dict, List, Optional, Union, Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

logger = logging.getLogger("plotter")


class DataLoader:
    """负责从各种来源加载数据"""

    @staticmethod
    def read_csv(file_path: str) -> tuple:
        """从CSV文件读取数据"""
        x = []
        y = []
        try:
            with open(file_path, "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    if row:  # Skip empty rows
                        x.append(int(row[0]))
                        y.append(float(row[2]))
            logger.debug(f"从 {file_path} 读取了 {len(x)} 个数据点")
        except Exception as e:
            logger.error(f"读取CSV文件 {file_path} 时出错: {str(e)}")
        return x, y

    @staticmethod
    def load_experiment_data(
        experiment_dirs: Union[str, List[str]], filename: str = "evaluation.csv"
    ) -> tuple:
        """加载一个或多个实验目录中的数据"""
        if isinstance(experiment_dirs, str):
            experiment_dirs = [experiment_dirs]

        all_x = []
        all_y = []

        for exp_dir in experiment_dirs:
            # 支持两种模式：
            # 1. 目录名包含通配符，如 "/path/to/exp*"
            # 2. 指定目录内的特定文件，如 "/path/to/exp/evaluation.csv"
            if "*" in exp_dir:
                # 目录名包含通配符
                matching_dirs = glob.glob(exp_dir)
                logger.info(
                    f"通配符路径 '{exp_dir}' 匹配到 {len(matching_dirs)} 个目录"
                )
                if not matching_dirs:
                    logger.warning(f"警告: 未找到匹配的目录: {exp_dir}")
                    continue

                for matching_dir in matching_dirs:
                    if os.path.isdir(matching_dir):
                        file_path = os.path.join(matching_dir, filename)
                        if os.path.exists(file_path):
                            logger.debug(f"处理文件: {file_path}")
                            x, y = DataLoader.read_csv(file_path)
                            all_x.append(x)
                            all_y.append(y)
                        else:
                            logger.debug(f"文件不存在: {file_path}")
            else:
                # 直接指定目录
                file_path = os.path.join(exp_dir, filename)
                file_paths = glob.glob(file_path)
                logger.info(f"路径 '{file_path}' 匹配到 {len(file_paths)} 个文件")

                for path in file_paths:
                    if os.path.exists(path):
                        logger.debug(f"处理文件: {path}")
                        x, y = DataLoader.read_csv(path)
                        all_x.append(x)
                        all_y.append(y)
                    else:
                        logger.debug(f"文件不存在: {path}")

        return all_x, all_y


class PlotStyle:
    """处理绘图样式设置"""

    @staticmethod
    def apply_style(cfg: DictConfig) -> None:
        """应用全局样式设置"""
        if hasattr(cfg, "figsize"):
            plt.figure(figsize=cfg.figsize)

        if hasattr(cfg, "font_size"):
            plt.rcParams.update({"font.size": cfg.font_size})

        if hasattr(cfg, "style") and cfg.style:
            plt.style.use(cfg.style)


class PlotRenderer:
    """负责渲染和保存图表"""

    @staticmethod
    def setup_plot(cfg: DictConfig) -> None:
        """设置基本的图表参数"""
        if hasattr(cfg, "title"):
            plt.title(cfg.title)

        if hasattr(cfg, "xlabel"):
            plt.xlabel(cfg.xlabel)

        if hasattr(cfg, "ylabel"):
            plt.ylabel(cfg.ylabel)

        if hasattr(cfg, "ylim"):
            plt.ylim(cfg.ylim)

        if hasattr(cfg, "xlim"):
            plt.xlim(cfg.xlim)

        if hasattr(cfg, "legend") and cfg.legend:
            plt.legend()

    @staticmethod
    def save_or_show(cfg: DictConfig) -> None:
        """根据配置保存或显示图表"""
        render_mode = cfg.get("render_mode", "plot")
        save_path = cfg.get("save_path", "./plots")

        # 确保保存目录存在
        if render_mode != "plot" and not os.path.exists(save_path):
            logger.info(f"创建保存目录: {save_path}")
            os.makedirs(save_path)

        if render_mode == "plot":
            logger.info("显示图表")
            plt.show()
        elif render_mode in ["svg", "png", "pdf"]:
            file_path = f"{save_path}/{cfg.save_name}.{render_mode}"
            logger.info(f"保存图表到: {file_path}")
            plt.savefig(file_path, dpi=cfg.get("dpi", 300), bbox_inches="tight")
            plt.close()
        else:
            error_msg = f"无效的渲染模式: {render_mode}，应为 plot/svg/png/pdf"
            logger.error(error_msg)
            raise ValueError(error_msg)


class PlotTypes:
    """包含各种类型的绘图函数"""

    @staticmethod
    def line_plot(x: List, y: List, cfg: DictConfig) -> None:
        """绘制基本的折线图"""
        plt.plot(
            x,
            y,
            color=cfg.get("color", "blue"),
            label=cfg.get("label", None),
            linestyle=cfg.get("linestyle", "-"),
            linewidth=cfg.get("linewidth", 1.5),
            marker=cfg.get("marker", None),
            markersize=cfg.get("markersize", 5),
        )

    @staticmethod
    def fill_between(x: List, y_min: List, y_max: List, cfg: DictConfig) -> None:
        """添加填充区域"""
        plt.fill_between(
            x, y_min, y_max, color=cfg.get("color", "blue"), alpha=cfg.get("alpha", 0.1)
        )

    @staticmethod
    def scatter_plot(x: List, y: List, cfg: DictConfig) -> None:
        """绘制散点图"""
        plt.scatter(
            x,
            y,
            color=cfg.get("color", "blue"),
            label=cfg.get("label", None),
            marker=cfg.get("marker", "o"),
            s=cfg.get("size", 50),
            alpha=cfg.get("alpha", 1.0),
            zorder=cfg.get("zorder", 2),
        )

    @staticmethod
    def horizontal_line(y_value: float, cfg: DictConfig) -> None:
        """绘制水平参考线"""
        x_range = plt.xlim()
        plt.plot(
            x_range,
            [y_value, y_value],
            color=cfg.get("color", "black"),
            linestyle=cfg.get("linestyle", "--"),
            label=cfg.get("label", None),
            linewidth=cfg.get("linewidth", 1.0),
        )

    @staticmethod
    def vertical_line(x_value: float, cfg: DictConfig) -> None:
        """绘制垂直参考线"""
        y_range = plt.ylim()
        plt.plot(
            [x_value, x_value],
            y_range,
            color=cfg.get("color", "black"),
            linestyle=cfg.get("linestyle", "--"),
            label=cfg.get("label", None),
            linewidth=cfg.get("linewidth", 1.0),
        )


class PlotRegistry:
    """注册和管理绘图类型"""

    _plot_functions: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """装饰器，用于注册绘图函数"""

        def decorator(func: Callable) -> Callable:
            cls._plot_functions[name] = func
            return func

        return decorator

    @classmethod
    def get_plot_function(cls, name: str) -> Callable:
        """获取注册的绘图函数"""
        if name not in cls._plot_functions:
            raise ValueError(f"Plot type '{name}' not registered")
        return cls._plot_functions[name]


class PlotManager:
    """主要的绘图管理器类"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.data_loader = DataLoader()

    def execute(self) -> None:
        """执行绘图过程"""
        # 应用全局样式
        logger.info("应用图表样式")
        PlotStyle.apply_style(self.cfg)

        # 获取指定的绘图函数
        plot_type = self.cfg.get("plot_type", "experiment_comparison")
        logger.info(f"使用绘图类型: {plot_type}")
        try:
            plot_func = PlotRegistry.get_plot_function(plot_type)
        except ValueError as e:
            logger.error(f"获取绘图函数失败: {str(e)}")
            raise

        # 执行绘图
        logger.info(f"开始执行 {plot_type} 绘图")
        plot_func(self.cfg)

        # 设置图表并保存/显示
        logger.info("设置图表属性")
        if (
            plot_type != "exception_rate"
        ):  # 跳过对异常率图的自动设置，因为它已经在函数内部设置了
            PlotRenderer.setup_plot(self.cfg)

        logger.info("保存/显示图表")
        PlotRenderer.save_or_show(self.cfg)
        logger.info("绘图过程完成")


# 注册各种绘图类型
@PlotRegistry.register("experiment_comparison")
def plot_experiment_comparison(cfg: DictConfig) -> None:
    """比较多个实验的结果"""
    experiment_dirs = cfg.experiment_dirs
    colors = cfg.get("colors", ["blue"] * len(experiment_dirs))
    labels = cfg.get(
        "labels", [f"Experiment {i + 1}" for i in range(len(experiment_dirs))]
    )
    filename = cfg.get("filename", "evaluation.csv")

    # 跟踪是否找到了有效数据
    data_found = False
    x_values = None

    for exp_dir, color, label in zip(experiment_dirs, colors, labels):
        # 处理包含通配符的路径
        if "*" in exp_dir:
            matching_dirs = glob.glob(exp_dir)
            logger.info(f"通配符路径 '{exp_dir}' 匹配到 {len(matching_dirs)} 个目录")
            if not matching_dirs:
                logger.warning(f"警告: 未找到匹配的目录: {exp_dir}")
                continue

            all_x = []
            all_y = []

            for matching_dir in matching_dirs:
                if os.path.isdir(matching_dir):
                    file_path = os.path.join(matching_dir, filename)
                    logger.debug(f"检查文件: {file_path}")
                    if os.path.exists(file_path):
                        logger.debug(f"处理文件: {file_path}")
                        x, y = DataLoader.read_csv(file_path)
                        # 确保x和y是相同长度
                        min_len = min(len(x), len(y))
                        logger.debug(
                            f"文件 {file_path} 包含 {len(x)} 个x值和 {len(y)} 个y值，使用前 {min_len} 个"
                        )
                        all_x.append(x[:min_len])
                        all_y.append(y[:min_len])
                        if not data_found and min_len > 0:
                            x_values = x[:min_len]
                            data_found = True
                    else:
                        logger.debug(f"文件不存在: {file_path}")
        else:
            # 直接指定目录
            file_path = os.path.join(exp_dir, filename)
            file_paths = glob.glob(file_path)
            all_x = []
            all_y = []

            for path in file_paths:
                if os.path.exists(path):
                    x, y = DataLoader.read_csv(path)
                    # 确保x和y是相同长度
                    min_len = min(len(x), len(y))
                    all_x.append(x[:min_len])
                    all_y.append(y[:min_len])
                    if not data_found and min_len > 0:
                        x_values = x[:min_len]
                        data_found = True

        if all_y:
            # 确保所有数据具有相同的长度以进行平均
            min_length = min(len(y) for y in all_y)
            logger.info(
                f"对于 {label}，找到 {len(all_y)} 个序列，最小长度为 {min_length}"
            )

            aligned_y = [y[:min_length] for y in all_y]
            aligned_x = all_x[0][:min_length] if all_x else []

            if aligned_x and aligned_y:
                logger.debug(f"计算 {len(aligned_y)} 个序列的平均值")
                average_y = np.mean(aligned_y, axis=0)

                # 绘制平均线
                logger.debug(f"绘制 {label} 的平均线")
                PlotTypes.line_plot(
                    aligned_x, average_y, DictConfig(dict(color=color, label=label))
                )

                # 添加填充区域
                if cfg.get("show_range", True) and len(aligned_y) > 1:
                    logger.debug(f"为 {label} 添加填充区域")
                    min_vals = np.min(aligned_y, axis=0)
                    max_vals = np.max(aligned_y, axis=0)
                    PlotTypes.fill_between(
                        aligned_x,
                        min_vals,
                        max_vals,
                        DictConfig(dict(color=color, alpha=0.1)),
                    )
                    logger.debug(
                        f"Y值范围: {min_vals[0]:.2f}-{max_vals[0]:.2f} (开始), "
                        f"{min_vals[-1]:.2f}-{max_vals[-1]:.2f} (结束)"
                    )

    # 如果没有找到任何数据，提前返回
    if not data_found:
        logger.warning("警告: 未找到任何有效数据")
        return

    # 添加基准线（如果指定）
    if "baseline_value" in cfg:
        PlotTypes.horizontal_line(
            cfg.baseline_value,
            DictConfig(dict(color="black", linestyle="--", label="Baseline")),
        )

    # 添加离线点（如果指定）
    if "offline_value" in cfg and x_values is not None and len(x_values) > 0:
        PlotTypes.scatter_plot(
            [x_values[0]],
            [cfg.offline_value],
            DictConfig(
                dict(color="black", marker="*", size=100, label="Offline (CQL)")
            ),
        )


@PlotRegistry.register("conbined_plot")
def plot_conbined_plot(cfg: DictConfig) -> None:
    """比较多个实验的结果"""
    experiment_dirs = cfg.experiment_dirs
    offline_experiment_dirs = cfg.get("offline_experiment_dirs", [])
    colors = cfg.get("colors", ["blue"] * len(experiment_dirs))
    labels = cfg.get(
        "labels", [f"Experiment {i + 1}" for i in range(len(experiment_dirs))]
    )
    filename = cfg.get("filename", "evaluation.csv")

    # 跟踪是否找到了有效数据
    data_found = False
    x_values = None

    for exp_dir, color, label in zip(experiment_dirs, colors, labels):
        # 处理包含通配符的路径
        if "*" in exp_dir:
            matching_dirs = glob.glob(exp_dir)
            logger.info(f"通配符路径 '{exp_dir}' 匹配到 {len(matching_dirs)} 个目录")
            if not matching_dirs:
                logger.warning(f"警告: 未找到匹配的目录: {exp_dir}")
                continue

            all_x = []
            all_y = []

            for matching_dir in matching_dirs:
                if os.path.isdir(matching_dir):
                    file_path = os.path.join(matching_dir, filename)
                    logger.debug(f"检查文件: {file_path}")
                    if os.path.exists(file_path):
                        logger.debug(f"处理文件: {file_path}")
                        x, y = DataLoader.read_csv(file_path)
                        # 确保x和y是相同长度
                        min_len = min(len(x), len(y))
                        logger.debug(
                            f"文件 {file_path} 包含 {len(x)} 个x值和 {len(y)} 个y值，使用前 {min_len} 个"
                        )
                        all_x.append(x[:min_len])
                        all_y.append(y[:min_len])
                        if not data_found and min_len > 0:
                            x_values = x[:min_len]
                            data_found = True
                    else:
                        logger.debug(f"文件不存在: {file_path}")
        else:
            # 直接指定目录
            file_path = os.path.join(exp_dir, filename)
            file_paths = glob.glob(file_path)
            all_x = []
            all_y = []

            for path in file_paths:
                if os.path.exists(path):
                    x, y = DataLoader.read_csv(path)
                    # 确保x和y是相同长度
                    min_len = min(len(x), len(y))
                    all_x.append(x[:min_len])
                    all_y.append(y[:min_len])
                    if not data_found and min_len > 0:
                        x_values = x[:min_len]
                        data_found = True

        if all_y:
            # 确保所有数据具有相同的长度以进行平均
            min_length = min(len(y) for y in all_y)
            logger.info(
                f"对于 {label}，找到 {len(all_y)} 个序列，最小长度为 {min_length}"
            )

            aligned_y = [y[:min_length] for y in all_y]
            aligned_x = all_x[0][:min_length] if all_x else []

            if aligned_x and aligned_y:
                logger.debug(f"计算 {len(aligned_y)} 个序列的平均值")
                average_y = np.mean(aligned_y, axis=0)

                # 绘制平均线
                logger.debug(f"绘制 {label} 的平均线")
                PlotTypes.line_plot(
                    aligned_x, average_y, DictConfig(dict(color=color, label=label))
                )

                # 添加填充区域
                if cfg.get("show_range", True) and len(aligned_y) > 1:
                    logger.debug(f"为 {label} 添加填充区域")
                    min_vals = np.min(aligned_y, axis=0)
                    max_vals = np.max(aligned_y, axis=0)
                    PlotTypes.fill_between(
                        aligned_x,
                        min_vals,
                        max_vals,
                        DictConfig(dict(color=color, alpha=0.1)),
                    )
                    logger.debug(
                        f"Y值范围: {min_vals[0]:.2f}-{max_vals[0]:.2f} (开始), "
                        f"{min_vals[-1]:.2f}-{max_vals[-1]:.2f} (结束)"
                    )

    # 如果没有找到任何数据，提前返回
    if not data_found:
        logger.warning("警告: 未找到任何有效数据")
        return

    # 添加基准线（如果指定）
    if "baseline_value" in cfg:
        PlotTypes.horizontal_line(
            cfg.baseline_value,
            DictConfig(dict(color="black", linestyle="--", label="Baseline")),
        )

    # 添加离线点（如果指定）
    if "offline_value" in cfg and x_values is not None and len(x_values) > 0:
        PlotTypes.scatter_plot(
            [x_values[0]],
            [cfg.offline_value],
            DictConfig(
                dict(color="black", marker="*", size=100, label="Offline (CQL)")
            ),
        )


@PlotRegistry.register("rollout_evaluation")
def plot_rollout_evaluation(cfg: DictConfig) -> None:
    """绘制rollout和evaluation对比图"""
    experiment_dir = cfg.experiment_dir
    colors = cfg.get("colors", ["blue", "red"])
    labels = cfg.get("labels", ["Rollout", "Evaluation"])
    filenames = cfg.get("filenames", ["rollout_return.csv", "evaluation.csv"])

    # 处理包含通配符的路径
    matching_dirs = [experiment_dir]
    if "*" in experiment_dir:
        matching_dirs = glob.glob(experiment_dir)
        if not matching_dirs:
            print(f"警告: 未找到匹配的目录: {experiment_dir}")
            return

    for filename, color, label in zip(filenames, colors, labels):
        all_x = []
        all_y = []

        for matching_dir in matching_dirs:
            if os.path.isdir(matching_dir):
                file_path = os.path.join(matching_dir, filename)
                if os.path.exists(file_path):
                    x, y = DataLoader.read_csv(file_path)
                    # 确保x和y是相同长度
                    min_len = min(len(x), len(y))
                    all_x.append(x[:min_len])
                    all_y.append(y[:min_len])

        if all_y:
            # 确保所有数据具有相同的长度以进行平均
            min_length = min(len(y) for y in all_y)
            aligned_y = [y[:min_length] for y in all_y]
            aligned_x = all_x[0][:min_length] if all_x else []

            if aligned_x and aligned_y:
                average_y = np.mean(aligned_y, axis=0)

                PlotTypes.line_plot(
                    aligned_x, average_y, DictConfig(dict(color=color, label=label))
                )

                if cfg.get("show_range", True) and len(aligned_y) > 1:
                    PlotTypes.fill_between(
                        aligned_x,
                        np.min(aligned_y, axis=0),
                        np.max(aligned_y, axis=0),
                        DictConfig(dict(color=color, alpha=0.1)),
                    )


@PlotRegistry.register("error_count")
def plot_error_count(cfg: DictConfig) -> None:
    """比较多个实验的结果"""
    experiment_dirs = cfg.experiment_dirs
    colors = cfg.get("colors", ["blue"] * len(experiment_dirs))
    labels = cfg.get(
        "labels", [f"Experiment {i + 1}" for i in range(len(experiment_dirs))]
    )
    filename = cfg.get("filename", "error_occurred.csv")

    # 跟踪是否找到了有效数据
    data_found = False
    x_values = None

    for exp_dir, color, label in zip(experiment_dirs, colors, labels):
        # 处理包含通配符的路径
        if "*" in exp_dir:
            matching_dirs = glob.glob(exp_dir)
            logger.info(f"通配符路径 '{exp_dir}' 匹配到 {len(matching_dirs)} 个目录")
            if not matching_dirs:
                logger.warning(f"警告: 未找到匹配的目录: {exp_dir}")
                continue

            all_x = []
            all_y = []

            for matching_dir in matching_dirs:
                if os.path.isdir(matching_dir):
                    file_path = os.path.join(matching_dir, filename)
                    logger.debug(f"检查文件: {file_path}")
                    if os.path.exists(file_path):
                        logger.debug(f"处理文件: {file_path}")
                        x, y = DataLoader.read_csv(file_path)
                        # 确保x和y是相同长度
                        min_len = min(len(x), len(y))
                        logger.debug(
                            f"文件 {file_path} 包含 {len(x)} 个x值和 {len(y)} 个y值，使用前 {min_len} 个"
                        )
                        all_x.append(x[:min_len])
                        all_y.append(y[:min_len])
                        if not data_found and min_len > 0:
                            x_values = x[:min_len]
                            data_found = True
                    else:
                        logger.debug(f"文件不存在: {file_path}")
        else:
            # 直接指定目录
            file_path = os.path.join(exp_dir, filename)
            file_paths = glob.glob(file_path)
            all_x = []
            all_y = []

            for path in file_paths:
                if os.path.exists(path):
                    x, y = DataLoader.read_csv(path)
                    # 确保x和y是相同长度
                    min_len = min(len(x), len(y))
                    all_x.append(x[:min_len])
                    all_y.append(y[:min_len])
                    if not data_found and min_len > 0:
                        x_values = x[:min_len]
                        data_found = True

        if all_y:
            for i, y in enumerate(all_y):
                for j in range(len(y)):
                    if j > 0:
                        all_y[i][j] += all_y[i][j - 1]  # 累加错误计数
            # 确保所有数据具有相同的长度以进行平均
            min_length = min(len(y) for y in all_y)
            logger.info(
                f"对于 {label}，找到 {len(all_y)} 个序列，最小长度为 {min_length}"
            )

            aligned_y = [y[:min_length] for y in all_y]
            aligned_x = all_x[0][:min_length] if all_x else []

            if aligned_x and aligned_y:
                logger.debug(f"计算 {len(aligned_y)} 个序列的平均值")
                average_y = np.mean(aligned_y, axis=0)

                # 绘制平均线
                logger.debug(f"绘制 {label} 的平均线")
                PlotTypes.line_plot(
                    aligned_x, average_y, DictConfig(dict(color=color, label=label))
                )

                # 添加填充区域
                if cfg.get("show_range", True) and len(aligned_y) > 1:
                    logger.debug(f"为 {label} 添加填充区域")
                    min_vals = np.min(aligned_y, axis=0)
                    max_vals = np.max(aligned_y, axis=0)
                    PlotTypes.fill_between(
                        aligned_x,
                        min_vals,
                        max_vals,
                        DictConfig(dict(color=color, alpha=0.1)),
                    )
                    logger.debug(
                        f"Y值范围: {min_vals[0]:.2f}-{max_vals[0]:.2f} (开始), "
                        f"{min_vals[-1]:.2f}-{max_vals[-1]:.2f} (结束)"
                    )

    # 如果没有找到任何数据，提前返回
    if not data_found:
        logger.warning("警告: 未找到任何有效数据")
        return

    # 添加基准线（如果指定）
    if "baseline_value" in cfg:
        PlotTypes.horizontal_line(
            cfg.baseline_value,
            DictConfig(
                dict(
                    color="black",
                    linestyle="--",
                    label=cfg.get("baseline_label", "Baseline"),
                )
            ),
        )

    # 添加离线点（如果指定）
    if "offline_value" in cfg and x_values is not None and len(x_values) > 0:
        PlotTypes.scatter_plot(
            [x_values[0]],
            [cfg.offline_value],
            DictConfig(
                dict(color="black", marker="*", size=100, label="Offline (CQL)")
            ),
        )


# @PlotRegistry.register('exception_rate')


@PlotRegistry.register("dual_data_plot")
def plot_dual_data(cfg: DictConfig) -> None:
    """结合两部分数据绘制在一张图表中

    配置参数:
    - pre_experiment_dirs: 从原点开始绘制的数据目录列表
    - pre_filename: 预先数据的文件名
    - pre_colors: 预先数据的颜色列表
    - experiment_dirs: 主要数据目录列表
    - x_offsets: 表示主要数据是接着pre_后绘制("pre")还是继续从原点绘制("ori")
    - vline_position: 代表分割两组数据的竖线位置
    - filename: 主要数据的文件名
    - colors: 主要数据的颜色列表
    - labels: 标签列表
    - show_range: 是否显示数据范围区域
    """
    # 处理预先数据部分 (pre_)
    pre_experiment_dirs = cfg.get("pre_experiment_dirs", [])
    pre_colors = cfg.get("pre_colors", ["blue"] * len(pre_experiment_dirs))
    pre_filename = cfg.get("pre_filename", "evaluation.csv")
    pre_labels = cfg.get(
        "pre_labels",
        [f"Pre Experiment {i + 1}" for i in range(len(pre_experiment_dirs))],
    )

    # 处理主要数据部分
    experiment_dirs = cfg.experiment_dirs
    colors = cfg.get("colors", ["red"] * len(experiment_dirs))
    labels = cfg.get(
        "labels", [f"Experiment {i + 1}" for i in range(len(experiment_dirs))]
    )
    filename = cfg.get("filename", "evaluation.csv")

    # 获取x偏移设置
    x_offsets = cfg.get("x_offsets", ["ori"] * len(experiment_dirs))

    # 跟踪是否找到了有效数据
    data_found = False
    max_pre_x = 0  # 预先数据的最大x值，用于连续绘制
    pre_data = []  # 存储预先数据的列表，用于后续参考
    pre_last_y = None  # 用于存储预先数据的最后一个y值

    # 首先处理预先数据
    for pre_exp_dir, pre_color, pre_label in zip(
        pre_experiment_dirs, pre_colors, pre_labels
    ):
        # 处理包含通配符的路径
        if "*" in pre_exp_dir:
            matching_dirs = glob.glob(pre_exp_dir)
            logger.info(
                f"通配符路径 '{pre_exp_dir}' 匹配到 {len(matching_dirs)} 个目录"
            )
            if not matching_dirs:
                logger.warning(f"警告: 未找到匹配的目录: {pre_exp_dir}")
                continue

            all_x = []
            all_y = []

            for matching_dir in matching_dirs:
                if os.path.isdir(matching_dir):
                    file_path = os.path.join(matching_dir, pre_filename)
                    if os.path.exists(file_path):
                        x, y = DataLoader.read_csv(file_path)
                        min_len = min(len(x), len(y))
                        all_x.append(x[:min_len])
                        all_y.append(y[:min_len])
                        if not data_found and min_len > 0:
                            data_found = True
        else:
            # 直接指定目录
            file_path = os.path.join(pre_exp_dir, pre_filename)
            file_paths = glob.glob(file_path)
            all_x = []
            all_y = []

            for path in file_paths:
                if os.path.exists(path):
                    x, y = DataLoader.read_csv(path)
                    min_len = min(len(x), len(y))
                    all_x.append(x[:min_len])
                    all_y.append(y[:min_len])
                    if not data_found and min_len > 0:
                        data_found = True

        if all_y:
            # 确保所有数据具有相同的长度以进行平均
            min_length = min(len(y) for y in all_y)

            aligned_y = [y[:min_length] for y in all_y]
            aligned_x = all_x[0][:min_length] if all_x else []

            if aligned_x and aligned_y:
                average_y = np.mean(aligned_y, axis=0)
                pre_last_y = average_y[-1] if len(average_y) > 0 else None

                # 绘制平均线
                PlotTypes.line_plot(
                    aligned_x,
                    average_y,
                    DictConfig(dict(color=pre_color, label=pre_label)),
                )

                # 添加填充区域
                if cfg.get("show_range", True) and len(aligned_y) > 1:
                    min_vals = np.min(aligned_y, axis=0)
                    max_vals = np.max(aligned_y, axis=0)
                    PlotTypes.fill_between(
                        aligned_x,
                        min_vals,
                        max_vals,
                        DictConfig(dict(color=pre_color, alpha=0.1, zorder=1)),
                    )

                # 更新最大x值
                if len(aligned_x) > 0:
                    max_pre_x = max(max_pre_x, aligned_x[-1])

                # 保存预先数据供后续使用
                pre_data.append(
                    {
                        "x": aligned_x,
                        "y": average_y,
                        "min_y": np.min(aligned_y, axis=0)
                        if len(aligned_y) > 1
                        else average_y,
                        "max_y": np.max(aligned_y, axis=0)
                        if len(aligned_y) > 1
                        else average_y,
                    }
                )

    # 绘制竖线分隔符（如果指定）
    if "vline_position" in cfg:
        PlotTypes.vertical_line(
            cfg.vline_position,
            DictConfig(dict(color="black", linestyle="--", label="_nolegend_")),
        )

    # 然后处理主要数据
    for idx, (exp_dir, color, label, x_offset) in enumerate(
        zip(experiment_dirs, colors, labels, x_offsets)
    ):
        # 处理包含通配符的路径
        if "*" in exp_dir:
            matching_dirs = glob.glob(exp_dir)
            logger.info(f"通配符路径 '{exp_dir}' 匹配到 {len(matching_dirs)} 个目录")
            if not matching_dirs:
                logger.warning(f"警告: 未找到匹配的目录: {exp_dir}")
                continue

            all_x = []
            all_y = []

            for matching_dir in matching_dirs:
                if os.path.isdir(matching_dir):
                    file_path = os.path.join(matching_dir, filename)
                    if os.path.exists(file_path):
                        x, y = DataLoader.read_csv(file_path)
                        min_len = min(len(x), len(y))
                        all_x.append(x[:min_len])
                        all_y.append(y[:min_len])
                        if not data_found and min_len > 0:
                            data_found = True
        else:
            # 直接指定目录
            file_path = os.path.join(exp_dir, filename)
            file_paths = glob.glob(file_path)
            all_x = []
            all_y = []

            for path in file_paths:
                if os.path.exists(path):
                    x, y = DataLoader.read_csv(path)
                    min_len = min(len(x), len(y))
                    all_x.append(x[:min_len])
                    all_y.append(y[:min_len])
                    if not data_found and min_len > 0:
                        data_found = True

        if all_y:
            # 确保所有数据具有相同的长度以进行平均
            min_length = min(len(y) for y in all_y)

            aligned_y = [y[:min_length] for y in all_y]
            aligned_x = all_x[0][:min_length] if all_x else []

            if aligned_x and aligned_y:
                average_y = np.mean(aligned_y, axis=0)
                if idx == 0 and pre_last_y is not None:
                    average_y[0] = pre_last_y  # 确保第一个点与预先数据对齐

                # 根据x_offset确定x轴偏移
                if x_offset == "pre" and max_pre_x > 0:
                    # 接着预先数据后绘制
                    offset_x = [x + max_pre_x for x in aligned_x]
                else:
                    # 从原点开始绘制
                    offset_x = aligned_x

                # 绘制平均线
                PlotTypes.line_plot(
                    offset_x, average_y, DictConfig(dict(color=color, label=label))
                )

                # 添加填充区域
                if cfg.get("show_range", True) and len(aligned_y) > 1:
                    min_vals = np.min(aligned_y, axis=0)
                    max_vals = np.max(aligned_y, axis=0)
                    PlotTypes.fill_between(
                        offset_x,
                        min_vals,
                        max_vals,
                        DictConfig(dict(color=color, alpha=0.1, zorder=1)),
                    )

    # 添加基准线（如果指定）
    if "baseline_value" in cfg:
        PlotTypes.horizontal_line(
            cfg.baseline_value,
            DictConfig(dict(color="black", linestyle="--", label="Baseline")),
        )

    # 添加离线点（如果指定）
    if "offline_value" in cfg and "offline_marker" in cfg:
        offline_marker = cfg.get("offline_marker", "*")
        offline_marker_size = cfg.get("offline_marker_size", 20)
        offline_marker_color = cfg.get("offline_marker_color", "black")
        offline_label = cfg.get("offline_label", "Offline")

        # 在图表开始位置绘制离线点
        PlotTypes.scatter_plot(
            [100],  # x位置在起始点
            # [cfg.offline_value],  # y值使用配置中的离线值
            [pre_last_y],  # y值使用配置中的离线值
            DictConfig(
                dict(
                    color=offline_marker_color,
                    marker=offline_marker,
                    size=offline_marker_size,
                    label=offline_label,
                    zorder=999,  # 确保离线点在最上层
                )
            ),
        )


@PlotRegistry.register("custom")
def plot_custom(cfg: DictConfig) -> None:
    """自定义绘图，直接使用配置中的函数调用"""
    # 这是一个通用的绘图函数，它将使用配置中指定的绘图调用
    # 配置应该包含绘图指令列表，每个指令指定要调用的函数和参数

    for instruction in cfg.instructions:
        func_name = instruction.function
        params = instruction.params

        # 获取绘图函数
        if hasattr(PlotTypes, func_name):
            func = getattr(PlotTypes, func_name)
            func(**params)
        else:
            logger.warning(f"未知的绘图函数: {func_name}")
            print(f"Warning: Function {func_name} not found in PlotTypes")


# 添加更多注册的绘图类型...
