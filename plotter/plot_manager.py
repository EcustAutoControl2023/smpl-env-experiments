import csv
import glob
import os
import logging
from typing import Dict, List, Union, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
            experiments_number = cfg.get("experiments_number", "")
            if experiments_number:
                file_path = f"{save_path}/{cfg.save_name}_exn-{experiments_number}.{render_mode}"
            else:
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
            x,
            y_min,
            y_max,
            color=cfg.get("color", "blue"),
            alpha=cfg.get("alpha", 0.1),
            linewidth=cfg.get("linewidth", 1.0),
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

    @staticmethod
    def triangular_heatmap(data: pd.DataFrame, cfg: DictConfig) -> None:
        """绘制三角热力图用于敏感性分析"""
        metric = cfg.get("metric", "combined_score")
        param1 = cfg.get("param1", "lambda_start")  # x-axis parameter
        param2 = cfg.get("param2", "lambda_end")  # y-axis parameter
        fixed_params = cfg.get("fixed_params", {})

        # Filter data based on fixed parameters
        subset = data.copy()
        for param, value in fixed_params.items():
            subset = subset[np.abs(subset[param] - value) < 1e-6]

        if len(subset) == 0:
            logger.warning(f"No data found for fixed parameters: {fixed_params}")
            return

        # Aggregate over seeds if present
        if "seed" in subset.columns:
            pivot_data = subset.groupby([param1, param2])[metric].mean().reset_index()
        else:
            pivot_data = subset

        pivot_matrix = pivot_data.pivot(index=param2, columns=param1, values=metric)

        # Sort index and columns to ensure proper ordering
        pivot_matrix = pivot_matrix.sort_index(ascending=False)
        pivot_matrix = pivot_matrix[sorted(pivot_matrix.columns)]

        # Create mask for upper triangle (where end > start, which is invalid)
        mask = np.zeros_like(pivot_matrix, dtype=bool)
        for i, row_val in enumerate(pivot_matrix.index):
            for j, col_val in enumerate(pivot_matrix.columns):
                if row_val > col_val:  # Invalid region
                    mask[i, j] = True

        # Plot heatmap
        sns.heatmap(
            pivot_matrix,
            mask=mask,
            annot=cfg.get("annotate", True),
            fmt=cfg.get("format", ".3f"),
            cmap=cfg.get("colormap", "viridis"),
            cbar_kws={"label": metric},
            vmin=cfg.get("vmin", float(np.nanmin(pivot_matrix.values))),
            vmax=cfg.get("vmax", float(np.nanmax(pivot_matrix.values))),
        )

        # Add diagonal line to show boundary
        # n = len(pivot_matrix)
        # plt.plot([n, 0], [0, n], "k--", alpha=0.3, linewidth=2)

        # Add text to explain the blank region
        plt.text(
            0.25,
            0.75,
            "end > start",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
            fontsize=24,
            alpha=0.5,
            rotation=45,
        )

    @staticmethod
    def sensitivity_comparison(data: pd.DataFrame, cfg: DictConfig) -> None:
        """绘制不同参数设置下的学习曲线对比"""
        combinations = list(cfg.get("combinations", []))
        x_col = cfg.get("x_column", "steps")
        y_cols = list(cfg.get("y_columns", ["returns", "errors"]))

        if len(combinations) == 0:
            logger.warning("No parameter combinations specified for comparison")
            return

        n_plots = len(y_cols)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for combo in combinations:
            # Filter data for this combination
            subset = data.copy()
            combo_dict = dict(combo) if hasattr(combo, "get") else combo
            params_dict = dict(combo_dict.get("params", {}))
            for param, value in params_dict.items():
                subset = subset[np.abs(subset[param] - value) < 1e-6]

            if len(subset) == 0:
                continue

            # Aggregate over seeds if present
            if "seed" in subset.columns:
                grouped = subset.groupby(x_col)
                for i, y_col in enumerate(y_cols):
                    mean_y = grouped[y_col].mean()
                    std_y = grouped[y_col].std()

                    axes[i].plot(
                        mean_y.index,
                        mean_y.values,
                        label=combo_dict.get("label", "Unknown"),
                        color=combo_dict.get("color", None),
                        linewidth=cfg.get("linewidth", 1.5),
                    )

                    if cfg.get("show_confidence", True):
                        axes[i].fill_between(
                            mean_y.index,
                            mean_y.values - std_y.values,
                            mean_y.values + std_y.values,
                            alpha=cfg.get("alpha", 0.2),
                            color=combo_dict.get("color", None),
                        )
            else:
                # Single trajectory
                for i, y_col in enumerate(y_cols):
                    axes[i].plot(
                        subset[x_col],
                        subset[y_col],
                        label=combo_dict.get("label", "Unknown"),
                        color=combo_dict.get("color", None),
                        linewidth=cfg.get("linewidth", 1.5),
                    )

        # Configure axes
        for i, y_col in enumerate(y_cols):
            axes[i].set_xlabel(cfg.get("xlabel", "Training Steps"))
            axes[i].set_ylabel(cfg.get("ylabel", {}).get(y_col, y_col))
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

    @staticmethod
    def sensitivity_metrics_table(data: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
        """创建敏感性分析的汇总表格"""
        metric = cfg.get("metric", "combined_score")
        group_by_params = list(cfg.get("group_by", ["lambda_start", "lambda_end"]))
        fixed_param_sets = list(cfg.get("fixed_param_sets", []))

        summary_data = []

        for param_set in fixed_param_sets:
            # Filter data for this parameter set
            subset = data.copy()
            param_dict = dict(param_set) if hasattr(param_set, "items") else param_set
            for param, value in param_dict.items():
                subset = subset[np.abs(subset[param] - value) < 1e-6]

            if len(subset) == 0:
                continue

            # Group by the varying parameters and compute statistics
            if "seed" in subset.columns:
                grouped = subset.groupby(group_by_params)[metric].agg(["mean", "std"])
            else:
                grouped = subset.groupby(group_by_params)[metric].agg(["mean"])
                grouped = grouped.to_frame()
                grouped["std"] = 0

            # Find best configuration
            best_idx = grouped["mean"].idxmax()
            best_mean = grouped.loc[best_idx, "mean"]
            best_std = grouped.loc[best_idx, "std"] if "std" in grouped.columns else 0

            # Create summary entry
            summary_entry = dict(param_dict)  # Convert to regular dict
            if isinstance(best_idx, tuple):
                for i, param in enumerate(group_by_params):
                    summary_entry[f"Best_{param}"] = float(
                        best_idx[i]
                    )  # Convert to Python float
            else:
                summary_entry[f"Best_{group_by_params[0]}"] = float(
                    best_idx
                )  # Convert to Python float

            summary_entry["Score"] = f"{float(best_mean):.3f} ± {float(best_std):.3f}"
            summary_data.append(summary_entry)

        return pd.DataFrame(summary_data)

    @staticmethod
    def compute_sensitivity_metrics(
        returns: np.ndarray, errors: np.ndarray, cfg: DictConfig
    ) -> Dict[str, float]:
        """计算敏感性分析指标"""
        eval_freq = cfg.get("eval_freq", 1000)
        n_steps = len(returns) * eval_freq
        steps = np.arange(len(returns)) * eval_freq

        # 1. Area Under Curve (sample efficiency)
        auc_return = np.trapz(returns, steps) / n_steps
        auc_error = np.trapz(errors, steps) / n_steps

        # 2. Final performance (last 20% of training)
        final_window = max(1, len(returns) // 5)
        final_return = np.mean(returns[-final_window:])
        final_return_std = np.std(returns[-final_window:])
        final_error = np.mean(errors[-final_window:])
        final_error_std = np.std(errors[-final_window:])

        # 3. Learning speed (steps to reach 90% of final performance)
        target_perf = 0.9 * final_return if final_return > 0 else 0
        speed_idx = np.where(returns >= target_perf)[0]
        speed_return = speed_idx[0] * eval_freq if len(speed_idx) > 0 else n_steps

        # 4. Stability (std dev in last 20% of training)
        stability_return = np.std(returns[-final_window:])
        stability_error = np.std(errors[-final_window:])

        # 5. Combined score
        return_range = returns.max() - returns.min()
        error_range = errors.max() - errors.min()

        if return_range > 0:
            norm_final_return = (final_return - returns.min()) / return_range
        else:
            norm_final_return = 1.0

        if error_range > 0:
            norm_final_error = 1 - (final_error - errors.min()) / error_range
        else:
            norm_final_error = 1.0

        # combined_score = np.sqrt(norm_final_return * norm_final_error)
        # norm_final_pref = np.sqrt(norm_final_return * norm_final_error)
        norm_final_pref = 0.5 * norm_final_return + 0.5 * norm_final_error

        if return_range > 0:
            norm_auc_score = (auc_return - returns.min()) / return_range
        else:
            norm_auc_score = 1.0
        if speed_return > 0:
            norm_speed_score = (speed_return - 0) / n_steps
        else:
            norm_speed_score = 1.0
        if stability_return > 0:
            norm_stability_score = (stability_return - 0) / (
                returns.max() - returns.min()
            )
        else:
            norm_stability_score = 1.0

        # 更全面的组合评分示例
        weights = {
            "final_perf": 1.0,  # 最终性能权重
            "efficiency": 0.0,  # 样本效率权重
            "speed": 0.0,  # 学习速度权重
            "stability": 0.0,  # 稳定性权重
        }
        # weights = {
        #     "final_perf": 2.0,  # 最终性能权重
        #     "efficiency": 4.0,  # 样本效率权重
        #     "speed": 2.0,  # 学习速度权重
        #     "stability": 2.0,  # 稳定性权重
        # }

        combined_score = (
            weights["final_perf"] * norm_final_pref
            + weights["efficiency"] * norm_auc_score
            + weights["speed"] * norm_speed_score
            + weights["stability"] * norm_stability_score
        )

        return {
            "auc_return": float(auc_return),
            "auc_error": float(auc_error),
            "final_return": float(final_return),
            "final_error": float(final_error),
            "speed_return": float(speed_return),
            "stability_return": float(stability_return),
            "stability_error": float(stability_error),
            cfg.get("metric", "combined_score"): float(combined_score),
        }

    @staticmethod
    def load_experimental_sensitivity_data(
        base_dir: str, pattern: str, cfg: DictConfig
    ) -> pd.DataFrame:
        """从实验目录加载敏感性分析数据"""
        import re
        import os

        results = []
        matching_dirs = glob.glob(os.path.join(base_dir, pattern))

        for exp_dir in matching_dirs:
            if not os.path.isdir(exp_dir):
                continue

            # 从目录名提取参数
            dir_name = os.path.basename(exp_dir)

            # 提取参数的正则表达式
            # 匹配 linears{s}e{e}k{k} 格式
            linear_match = re.search(r"linears(\d+)e(\d+)k(\d+)", dir_name)
            if not linear_match:
                continue

            s_param = int(linear_match.group(1))
            e_param = int(linear_match.group(2))
            k_param = int(linear_match.group(3))

            # 提取seed
            seed_match = re.search(r"seed-(\d+)", dir_name)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))

            # 读取evaluation.csv
            eval_path = os.path.join(exp_dir, "evaluation.csv")
            error_path = os.path.join(exp_dir, "error_occurred.csv")

            if not os.path.exists(eval_path) or not os.path.exists(error_path):
                continue

            try:
                # 读取evaluation数据
                eval_data = pd.read_csv(
                    eval_path, header=None, names=["episode", "step", "return"]
                )
                returns = eval_data["return"].values
                steps = eval_data["step"].values

                # 读取error数据
                error_data = pd.read_csv(
                    error_path, header=None, names=["episode", "step", "error"]
                )
                errors = error_data["error"].values

                # 处理数组长度不匹配的问题
                min_length = min(len(returns), len(errors))
                if min_length == 0:
                    continue

                returns = returns[:min_length]
                errors = errors[:min_length]

                # 计算metrics
                metrics = PlotTypes.compute_sensitivity_metrics(returns, errors, cfg)

                # 存储结果
                result = {
                    "lambda_start": s_param / 100.0,  # 归一化参数
                    "lambda_end": e_param / 100.0,
                    "eps_0": k_param,  # k参数作为eps_0
                    "eps_T": 0.0,  # 固定eps_T为0
                    "seed": seed,
                    **metrics,
                }
                results.append(result)

            except Exception as e:
                logger.warning(f"处理实验目录 {exp_dir} 时出错: {str(e)}")
                continue

        if not results:
            logger.error(f"未找到匹配的实验数据: {pattern}")
            return pd.DataFrame()

        logger.info(f"成功加载 {len(results)} 个实验结果")
        return pd.DataFrame(results)

    def bar_plot(self, groups_data, cfg):
        """绘制柱状图

        Args:
            groups_data: 分组数据，格式为 {group_name: {metric_name: value}}
            cfg: 配置对象
        """
        import matplotlib.pyplot as plt

        group_names = list(groups_data.keys())
        metric_names = cfg.get("metric_names", ["Performance", "Error"])
        colors = cfg.get("colors", ["skyblue", "lightcoral"])

        # 检查是否启用百分比模式
        use_percentage = cfg.get("use_percentage", False)
        if use_percentage:
            groups_data = self._convert_to_percentage(groups_data, metric_names)

        # 检查绘图模式
        group_by_metric = cfg.get("group_by_metric", False)
        side_by_side = cfg.get("side_by_side", False)

        if side_by_side:
            # 侧并排模式：相同算法的配置紧贴，用颜色区分
            return self._bar_plot_side_by_side(
                groups_data, cfg, group_names, metric_names, colors
            )
        elif group_by_metric:
            # 按指标分组：每个指标的所有算法放在一起
            return self._bar_plot_grouped_by_metric(
                groups_data, cfg, group_names, metric_names, colors
            )
        else:
            # 按算法分组：每个算法的所有指标放在一起
            return self._bar_plot_grouped_by_algorithm(
                groups_data, cfg, group_names, metric_names, colors
            )

    def _convert_to_percentage(self, groups_data, metric_names):
        """将数据转换为百分比形式，每个指标的最高值为100%"""
        converted_data = {}

        # 为每个指标找到最大值
        max_values = {}
        for metric in metric_names:
            max_val = 0
            for group_data in groups_data.values():
                val = group_data.get(metric, 0)
                if val > max_val:
                    max_val = val
            max_values[metric] = max_val if max_val > 0 else 1  # 避免除零

        # 转换所有数据为百分比
        for group_name, group_data in groups_data.items():
            converted_data[group_name] = {}
            for metric in metric_names:
                original_value = group_data.get(metric, 0)
                percentage_value = (original_value / max_values[metric]) * 100
                converted_data[group_name][metric] = percentage_value

        return converted_data

    def _bar_plot_grouped_by_algorithm(
        self, groups_data, cfg, group_names, metric_names, colors
    ):
        """按算法分组的柱状图（原有逻辑）"""
        import matplotlib.pyplot as plt

        # 设置柱状图参数
        x = np.arange(len(group_names))
        width = cfg.get("bar_width", 0.35)

        fig, ax = plt.subplots(figsize=cfg.get("figsize", (10, 6)))

        # 为每个指标绘制柱状图
        for i, metric in enumerate(metric_names):
            values = [groups_data[group].get(metric, 0) for group in group_names]
            offset = (i - len(metric_names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, values, width, label=metric, color=colors[i % len(colors)]
            )

            # 在柱子上添加数值标签
            if cfg.get("show_values", True):
                use_percentage = cfg.get("use_percentage", False)
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    if use_percentage:
                        label_text = f"{value:.1f}%"
                    else:
                        label_text = f"{value:.2f}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(values) * 0.01,
                        label_text,
                        ha="center",
                        va="bottom",
                    )
                    # label_text, ha='center', va='bottom', fontsize=9)

        # 设置标签和标题
        ax.set_xlabel(cfg.get("xlabel", "Groups"))
        use_percentage = cfg.get("use_percentage", False)
        default_ylabel = "Percentage (%)" if use_percentage else "Values"
        ax.set_ylabel(cfg.get("ylabel", default_ylabel))
        ax.set_title(cfg.get("title", "Bar Chart Comparison"))
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=cfg.get("xlabel_rotation", 45))

        # 设置y轴范围
        if "ylim" in cfg:
            ax.set_ylim(cfg.ylim)

        # 显示图例
        if cfg.get("legend", True):
            ax.legend()

        # 调整布局
        plt.tight_layout()

        return fig, ax

    def _bar_plot_grouped_by_metric(
        self, groups_data, cfg, group_names, metric_names, colors
    ):
        """按指标分组的柱状图"""
        import matplotlib.pyplot as plt

        # 设置柱状图参数
        width = cfg.get("bar_width", 0.8)
        group_spacing = cfg.get("group_spacing", 1.5)  # 指标组之间的间距

        fig, ax = plt.subplots(figsize=cfg.get("figsize", (10, 6)))

        # 计算每个指标组的位置
        n_groups = len(group_names)
        n_metrics = len(metric_names)

        # 创建x轴位置
        metric_positions = []
        all_labels = []

        for i, metric in enumerate(metric_names):
            # 每个指标组的起始位置
            start_pos = i * (n_groups + group_spacing)
            positions = np.arange(start_pos, start_pos + n_groups)
            metric_positions.append(positions)

            # 收集该指标的所有数值
            values = [groups_data[group].get(metric, 0) for group in group_names]

            # 绘制该指标的所有柱子
            bars = ax.bar(
                positions, values, width, label=metric, color=colors[i % len(colors)]
            )

            # 在柱子上添加数值标签
            if cfg.get("show_values", True):
                use_percentage = cfg.get("use_percentage", False)
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    if use_percentage:
                        label_text = f"{value:.1f}%"
                    else:
                        label_text = f"{value:.2f}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(values) * 0.01,
                        label_text,
                        ha="center",
                        va="bottom",
                        fontsize=24,
                    )

        # 设置x轴标签
        all_positions = []
        all_labels = []

        for i, (metric, positions) in enumerate(zip(metric_names, metric_positions)):
            all_positions.extend(positions)
            all_labels.extend(group_names)

            # 在每个指标组下方添加指标名称
            center_pos = np.mean(positions)
            ax.text(
                center_pos,
                ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1,
                metric,
                ha="center",
                va="top",
                fontweight="bold",
                fontsize=24,
            )

        # 设置标签和标题
        ax.set_xlabel(cfg.get("xlabel", "Metrics"))
        use_percentage = cfg.get("use_percentage", False)
        default_ylabel = "Percentage (%)" if use_percentage else "Values"
        ax.set_ylabel(cfg.get("ylabel", default_ylabel))
        ax.set_title(cfg.get("title", "Bar Chart Comparison"))
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, rotation=cfg.get("xlabel_rotation", 45))

        # 设置y轴范围
        if "ylim" in cfg:
            ax.set_ylim(cfg.ylim)

        # 显示图例
        if cfg.get("legend", True):
            ax.legend()

        # 调整布局
        plt.tight_layout()

        return fig, ax

    def _bar_plot_side_by_side(
        self, groups_data, cfg, group_names, metric_names, colors
    ):
        """侧并排模式：相同算法的配置紧贴，用颜色区分配置"""
        import matplotlib.pyplot as plt

        # 解析算法和配置
        algorithms = []
        configs = []

        for group_name in group_names:
            if "(" in group_name and ")" in group_name:
                algorithm = group_name.split("(")[0].strip()
                config = group_name.split("(")[1].split(")")[0].strip()
                algorithms.append(algorithm)
                configs.append(config)
            else:
                algorithms.append(group_name)
                configs.append("default")

        # 获取唯一的算法和配置
        unique_algorithms = []
        for alg in algorithms:
            if alg not in unique_algorithms:
                unique_algorithms.append(alg)

        unique_configs = []
        for conf in configs:
            if conf not in unique_configs:
                unique_configs.append(conf)

        # 设置颜色映射
        config_colors = cfg.get("config_colors", {})
        if not config_colors:
            default_colors = ["#1f77b4", "#ff7f0e"]
            for i, config in enumerate(unique_configs):
                config_colors[config] = default_colors[i % len(default_colors)]

        # 组织数据
        data_by_algorithm = {}
        for algorithm in unique_algorithms:
            data_by_algorithm[algorithm] = {}
            for config in unique_configs:
                data_by_algorithm[algorithm][config] = {}

        # 填充数据
        for i, group_name in enumerate(group_names):
            algorithm = algorithms[i]
            config = configs[i]
            if algorithm in unique_algorithms and config in unique_configs:
                for metric in metric_names:
                    value = groups_data[group_name].get(metric, 0)
                    data_by_algorithm[algorithm][config][metric] = value

        # 创建子图
        n_metrics = len(metric_names)
        if n_metrics == 1:
            fig, ax = plt.subplots(figsize=cfg.get("figsize", (10, 6)))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_metrics, figsize=cfg.get("figsize", (16, 6)))

        width = cfg.get("bar_width", 0.35)
        x = range(len(unique_algorithms))

        print(metric_names)
        # assert False
        for metric_idx, metric in enumerate(metric_names):
            ax = axes[metric_idx]

            for config_idx, config in enumerate(unique_configs):
                values = []
                for alg in unique_algorithms:
                    if config in data_by_algorithm[alg]:
                        values.append(data_by_algorithm[alg][config].get(metric, 0))
                    else:
                        values.append(0)

                offset = (config_idx - 0.5) * width
                positions = [pos + offset for pos in x]

                bars = ax.bar(
                    positions,
                    values,
                    width,
                    label=config,
                    color=config_colors.get(config, "#666666"),
                    alpha=0.8,
                )

                # 添加数值标签
                if cfg.get("show_values", True):
                    use_percentage = cfg.get("use_percentage", False)
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        if height > 0:
                            if use_percentage:
                                label_text = f"{value:.1f}%"
                            else:
                                label_text = f"{value:.0f}"
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height + max(values) * 0.01,
                                label_text,
                                ha="center",
                                va="bottom",
                                fontsize=24,
                                fontweight="bold",
                            )

            ax.set_xlabel(
                cfg.get("xlabel", "Algorithms"),
                fontsize=24,
                fontweight="bold",
            )
            use_percentage = cfg.get("use_percentage", False)
            ylabel = f"{metric} (%)" if use_percentage else metric
            ax.set_ylabel(
                ylabel,
                fontsize=24,
                fontweight="bold",
            )
            ax.set_title(
                f"",
                fontsize=24,
                fontweight="bold",
            )
            # ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(unique_algorithms, fontsize=24)

            if cfg.get("legend", True) and metric_idx == 0:  # 只在第一个子图显示图例
                ax.legend(fontsize=24)
                # ax.legend()

            if cfg.get("grid", True):
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes[0] if n_metrics == 1 else axes

    def compute_performance_metrics(self, experiment_dirs, filename="evaluation.csv"):
        """计算实验的性能指标

        Args:
            experiment_dirs: 实验目录列表
            filename: 数据文件名

        Returns:
            dict: 包含平均性能等指标的字典
        """
        all_data = []

        for exp_dir in experiment_dirs:
            if "*" in exp_dir:
                matching_dirs = glob.glob(exp_dir)
                logger.info(
                    f"通配符路径 '{exp_dir}' 匹配到 {len(matching_dirs)} 个目录"
                )

                exp_data = []
                for matching_dir in matching_dirs:
                    if os.path.isdir(matching_dir):
                        file_path = os.path.join(matching_dir, filename)
                        if os.path.exists(file_path):
                            x, y = DataLoader.read_csv(file_path)
                            if len(y) > 0:
                                exp_data.append(y)

                if exp_data:
                    # 计算所有种子的平均性能
                    min_length = min(len(y) for y in exp_data)
                    aligned_data = [y[:min_length] for y in exp_data]
                    mean_performance = np.mean([np.mean(y) for y in aligned_data])
                    # 计算后20%的平均性能
                    last_20_percent = int(min_length * 0.2)
                    if last_20_percent > 0:
                        last_20_performance = np.mean(
                            [np.mean(y[-last_20_percent:]) for y in aligned_data]
                        )
                    else:
                        last_20_performance = mean_performance

                    all_data.append(
                        {
                            "mean_performance": mean_performance,
                            "last_20_performance": last_20_performance,
                            "data_count": len(exp_data),
                        }
                    )

        return all_data

    def compute_error_metrics(self, experiment_dirs, filename="error_count.csv"):
        """计算实验的错误指标

        Args:
            experiment_dirs: 实验目录列表
            filename: 错误数据文件名

        Returns:
            list: 包含错误统计的列表
        """
        all_errors = []

        for exp_dir in experiment_dirs:
            if "*" in exp_dir:
                matching_dirs = glob.glob(exp_dir)
                logger.info(
                    f"通配符路径 '{exp_dir}' 匹配到 {len(matching_dirs)} 个目录"
                )

                exp_errors = []
                for matching_dir in matching_dirs:
                    if os.path.isdir(matching_dir):
                        file_path = os.path.join(matching_dir, filename)
                        if os.path.exists(file_path):
                            try:
                                x, y = DataLoader.read_csv(file_path)
                                if len(y) > 0:
                                    total_errors = np.sum(y)
                                    exp_errors.append(total_errors)
                            except Exception as e:
                                logger.warning(f"无法读取错误文件 {file_path}: {e}")
                                exp_errors.append(0)
                        else:
                            # 如果错误文件不存在，假设错误为0
                            exp_errors.append(0)

                if exp_errors:
                    mean_errors = np.mean(exp_errors)
                    all_errors.append(mean_errors)
                else:
                    all_errors.append(0)

        return all_errors


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

    def execute(self, cfg: DictConfig = None) -> None:
        """执行绘图过程"""
        # 使用传入的cfg或者实例的cfg
        config = cfg if cfg is not None else self.cfg

        # 应用全局样式
        logger.info("应用图表样式")
        PlotStyle.apply_style(config)

        # 获取指定的绘图函数
        plot_type = config.get("plot_type", "experiment_comparison")
        logger.info(f"使用绘图类型: {plot_type}")
        try:
            plot_func = PlotRegistry.get_plot_function(plot_type)
        except ValueError as e:
            logger.error(f"获取绘图函数失败: {str(e)}")
            raise

        # 执行绘图
        logger.info(f"开始执行 {plot_type} 绘图")
        plot_func(config)

        # 设置图表并保存/显示
        logger.info("设置图表属性")
        if (
            plot_type != "exception_rate"
        ):  # 跳过对异常率图的自动设置，因为它已经在函数内部设置了
            PlotRenderer.setup_plot(config)

        logger.info("保存/显示图表")
        PlotRenderer.save_or_show(config)
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

        # 对all_y进行切片(cfg.get("experiments_number", len(all_y)))次
        if len(all_y) > 0:
            all_y = all_y[: cfg.get("experiments_number", len(all_y))]
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
                    DictConfig(
                        dict(
                            color=pre_color,
                            label=pre_label,
                            linewidth=4,
                            linestyle="--",
                        )
                    ),
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

        # 对all_y进行切片(cfg.get("experiments_number", len(all_y)))次
        if len(all_y) > 0:
            all_y = all_y[: cfg.get("experiments_number", len(all_y))]
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


@PlotRegistry.register("sensitivity_heatmap")
def plot_sensitivity_heatmap(cfg: DictConfig) -> None:
    """绘制敏感性分析热力图"""
    data_path = cfg.get("data_path")
    if not data_path:
        logger.error("未指定数据路径")
        return

    # 加载数据
    try:
        if data_path.endswith(".csv"):
            data = pd.read_csv(data_path)
        else:
            logger.error(f"不支持的数据格式: {data_path}")
            return
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return

    # 设置绘图
    renderer = PlotRenderer()
    renderer.setup_plot(cfg)

    # 绘制热力图
    PlotTypes.triangular_heatmap(data, cfg)

    # 应用样式
    style = PlotStyle()
    style.apply_style(cfg)

    # 保存或显示
    renderer.save_or_show(cfg)


@PlotRegistry.register("sensitivity_comparison")
def plot_sensitivity_comparison(cfg: DictConfig) -> None:
    """绘制敏感性分析对比图"""
    data_path = cfg.get("data_path")
    if not data_path:
        logger.error("未指定数据路径")
        return

    # 加载数据
    try:
        if data_path.endswith(".csv"):
            data = pd.read_csv(data_path)
        else:
            logger.error(f"不支持的数据格式: {data_path}")
            return
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return

    # 设置绘图
    renderer = PlotRenderer()
    renderer.setup_plot(cfg)

    # 绘制对比图
    PlotTypes.sensitivity_comparison(data, cfg)

    # 应用样式
    style = PlotStyle()
    style.apply_style(cfg)

    # 保存或显示
    renderer.save_or_show(cfg)


@PlotRegistry.register("sensitivity_summary")
def plot_sensitivity_summary(cfg: DictConfig) -> None:
    """生成敏感性分析汇总表格和图表"""
    data_path = cfg.get("data_path")
    if not data_path:
        logger.error("未指定数据路径")
        return

    # 加载数据
    try:
        if data_path.endswith(".csv"):
            data = pd.read_csv(data_path)
        else:
            logger.error(f"不支持的数据格式: {data_path}")
            return
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return

    # 生成汇总表格
    summary_table = PlotTypes.sensitivity_metrics_table(data, cfg)

    # 打印汇总表格
    print("\n=== Sensitivity Analysis Summary ===")
    print(summary_table.to_string(index=False))

    # 如果需要，保存为LaTeX格式
    if cfg.get("save_latex", False):
        latex_output = cfg.get("latex_output", "sensitivity_summary.tex")
        with open(latex_output, "w") as f:
            f.write(summary_table.to_latex(index=False))
        print(f"\nLaTeX table saved to: {latex_output}")

    # 如果需要，保存为CSV格式
    if cfg.get("save_csv", False):
        csv_output = cfg.get("csv_output", "sensitivity_summary.csv")
        summary_table.to_csv(csv_output, index=False)
        print(f"CSV table saved to: {csv_output}")


@PlotRegistry.register("generate_sensitivity_data")
def generate_sensitivity_data(cfg: DictConfig) -> None:
    """生成敏感性分析的模拟数据"""
    n_points = cfg.get("n_points", 5)
    n_seeds = cfg.get("n_seeds", 5)
    output_path = cfg.get("output_path", "sensitivity_results.csv")

    # 生成有效的参数网格
    params = []
    vals = np.linspace(0, 1, n_points)

    for lambda_start in vals:
        for lambda_end in vals:
            if lambda_end <= lambda_start:  # 有效约束
                for eps_start in vals:
                    for eps_end in vals:
                        if eps_end <= eps_start:  # 有效约束
                            params.append(
                                {
                                    "lambda_start": lambda_start,
                                    "lambda_end": lambda_end,
                                    "eps_0": eps_start,
                                    "eps_T": eps_end,
                                }
                            )

    results = []
    for param_combo in params:
        for seed in range(n_seeds):
            np.random.seed(seed)

            # 模拟训练数据
            n_steps = cfg.get("n_steps", 100000)
            eval_freq = cfg.get("eval_freq", 1000)
            n_evals = n_steps // eval_freq
            t = np.linspace(0, 1, n_evals)

            # 参数影响学习
            lambda_effect = (
                param_combo["lambda_start"] + param_combo["lambda_end"]
            ) / 2
            eps_effect = (param_combo["eps_0"] + param_combo["eps_T"]) / 2

            # 模拟学习曲线
            returns = 100 * (
                1 - np.exp(-3 * t * (1 + lambda_effect))
            ) + np.random.normal(0, 5, n_evals)
            errors = 50 * np.exp(-2 * t * (1 + eps_effect)) + np.random.normal(
                0, 2, n_evals
            )
            errors = np.maximum(0, np.cumsum(errors))  # 累积错误

            # 计算指标
            metrics = PlotTypes.compute_sensitivity_metrics(returns, errors, cfg)

            # 存储结果
            result = {**param_combo, **metrics, "seed": seed}
            results.append(result)

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(
        f"Generated {len(results_df)} sensitivity analysis results and saved to {output_path}"
    )


@PlotRegistry.register("experimental_sensitivity_analysis")
def plot_experimental_sensitivity_analysis(cfg: DictConfig) -> None:
    """基于真实实验数据的敏感性分析"""
    base_dir = cfg.get("base_dir")
    pattern = cfg.get("pattern", "*")

    if not base_dir:
        logger.error("未指定base_dir")
        return

    # 加载实验数据
    try:
        data = PlotTypes.load_experimental_sensitivity_data(base_dir, pattern, cfg)
        if data.empty:
            logger.error("未加载到有效的实验数据")
            return

        logger.info(f"成功加载 {len(data)} 个实验数据点")
        logger.info(
            f"参数范围: lambda_start=[{data['lambda_start'].min():.2f}, {data['lambda_start'].max():.2f}], "
            f"lambda_end=[{data['lambda_end'].min():.2f}, {data['lambda_end'].max():.2f}], "
            f"eps_0=[{data['eps_0'].min():.0f}, {data['eps_0'].max():.0f}]"
        )

    except Exception as e:
        logger.error(f"加载实验数据失败: {str(e)}")
        return

    # 设置绘图
    renderer = PlotRenderer()

    # 创建多个分析图表
    plot_types = cfg.get("analysis_types", ["heatmap", "summary"])

    if "heatmap" in plot_types:
        # 创建热力图
        heatmap_cfg = DictConfig(
            {
                "metric": cfg.get("metric", "combined_score"),
                "param1": cfg.get("param1", "lambda_start"),
                "param2": cfg.get("param2", "lambda_end"),
                "fixed_params": cfg.get("fixed_params", {"eps_0": 1}),
                "annotate": cfg.get("annotate", True),
                "format": cfg.get("format", ".3f"),
                "colormap": cfg.get("colormap", "viridis"),
            }
        )

        renderer.setup_plot(cfg)
        PlotTypes.triangular_heatmap(data, heatmap_cfg)

        # 设置标题和标签
        plt.title(cfg.get("title", "Parameter Sensitivity Analysis"))
        plt.xlabel(cfg.get("xlabel", "λ₀ (start)"))
        plt.ylabel(cfg.get("ylabel", "λ_T (end)"))

        # 应用样式
        style = PlotStyle()
        style.apply_style(cfg)

        # 保存或显示
        if cfg.get("save_heatmap", True):
            save_path = cfg.get("heatmap_output", "sensitivity_heatmap.pdf")
            plt.savefig(save_path, dpi=cfg.get("dpi", 300), bbox_inches="tight")
            logger.info(f"热力图已保存到: {save_path}")

        if cfg.get("show_plot", False):
            plt.show()

        # plt.clf()

    if "summary" in plot_types:
        # 生成汇总统计
        summary_cfg = DictConfig(
            {
                "metric": cfg.get("metric", "combined_score"),
                "group_by": cfg.get("group_by", ["lambda_start", "lambda_end"]),
                "fixed_param_sets": cfg.get("fixed_param_sets", [{"eps_0": 1}]),
            }
        )

        summary_table = PlotTypes.sensitivity_metrics_table(data, summary_cfg)

        # 打印汇总表格
        print("\n=== Experimental Sensitivity Analysis Summary ===")
        if not summary_table.empty:
            print(summary_table.to_string(index=False))

            # 保存表格
            if cfg.get("save_summary", True):
                csv_output = cfg.get(
                    "summary_csv", "experimental_sensitivity_summary.csv"
                )
                summary_table.to_csv(csv_output, index=False)
                logger.info(f"汇总表格已保存到: {csv_output}")

                if cfg.get("save_latex", False):
                    latex_output = cfg.get(
                        "latex_output", "experimental_sensitivity_summary.tex"
                    )
                    with open(latex_output, "w") as f:
                        f.write(summary_table.to_latex(index=False))
                    logger.info(f"LaTeX表格已保存到: {latex_output}")
        else:
            print("未生成汇总表格 - 数据可能不足")

    # 统计分析
    if "statistics" in plot_types:
        print("\n=== Parameter Effects Analysis ===")
        for metric in ["combined_score", "final_return", "final_error"]:
            if metric in data.columns:
                lambda_effect = data.groupby("lambda_start")[metric].mean().std()
                eps_effect = data.groupby("eps_0")[metric].mean().std()
                print(f"\n{metric}:")
                print(f"  Effect of λ₀: {lambda_effect:.4f}")
                print(f"  Effect of ε₀: {eps_effect:.4f}")
                if eps_effect > 0:
                    print(f"  Ratio λ/ε: {lambda_effect / eps_effect:.2f}")

        # 找到最佳参数组合
        print("\n=== Best Parameter Combinations ===")
        for metric in ["combined_score", "final_return"]:
            if metric in data.columns:
                best_idx = data[metric].idxmax()
                best_row = data.iloc[best_idx]
                print(f"\nBest {metric}:")
                print(
                    f"  λ₀={best_row['lambda_start']:.2f}, λ_T={best_row['lambda_end']:.2f}"
                )
                print(f"  ε₀={best_row['eps_0']:.0f}, ε_T={best_row['eps_T']:.0f}")
                print(f"  Score: {best_row[metric]:.4f}")

    logger.info("实验敏感性分析完成")


@PlotRegistry.register("bar_plot")
def plot_bar_plot(cfg: DictConfig) -> None:
    """绘制柱状图比较

    用于比较不同算法/方法的性能和错误率
    """
    experiment_dirs = cfg.experiment_dirs
    group_labels = cfg.get(
        "group_labels", [f"Group {i + 1}" for i in range(len(experiment_dirs))]
    )

    # 收集所有组的数据
    groups_data = {}

    for i, (exp_dir, group_label) in enumerate(zip(experiment_dirs, group_labels)):
        logger.info(f"处理组 {group_label}: {exp_dir}")

        # 计算性能指标
        performance_data = PlotTypes().compute_performance_metrics(
            [exp_dir], cfg.get("performance_filename", "evaluation.csv")
        )
        error_data = PlotTypes().compute_error_metrics(
            [exp_dir], cfg.get("error_filename", "error_count.csv")
        )

        if performance_data:
            # 使用后20%的平均性能
            performance_value = performance_data[0].get("last_20_performance", 0)
        else:
            performance_value = 0
            logger.warning(f"未找到组 {group_label} 的性能数据")

        if error_data:
            error_value = error_data[0]
        else:
            error_value = 0
            logger.warning(f"未找到组 {group_label} 的错误数据")

        # 存储数据
        groups_data[group_label] = {
            cfg.get(
                "performance_metric_name", "Performance (Last 20%)"
            ): performance_value,
            cfg.get("error_metric_name", "Total Errors"): error_value,
        }

        logger.info(
            f"组 {group_label} - 性能: {performance_value:.2f}, 错误: {error_value:.2f}"
        )

    if not groups_data:
        logger.error("未找到任何有效数据")
        return

    # 设置绘图
    renderer = PlotRenderer()
    renderer.setup_plot(cfg)

    # 绘制柱状图
    plot_types = PlotTypes()
    fig, ax = plot_types.bar_plot(groups_data, cfg)

    # 应用样式
    style = PlotStyle()
    style.apply_style(cfg)

    # 保存或显示
    renderer.save_or_show(cfg)


# 添加更多注册的绘图类型...
