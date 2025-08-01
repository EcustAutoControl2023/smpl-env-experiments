#!/usr/bin/env python3
"""
绘图系统命令行工具

这是一个方便的命令行工具，用于直接调用绘图系统而不需要使用完整的Hydra命令。
它提供了一系列常用的绘图命令，并且可以轻松指定实验目录和其他参数。

使用方法:
    python -m plotter.plot experiment_comparison --dirs path1 path2 --title "My Plot"
    python -m plotter.plot rollout_evaluation --dir path1 --title "Rollout vs Eval"
"""
import argparse
import sys
import os
import logging
from typing import List, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('plotter.cli')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_common_args(parser):
    """添加所有绘图类型通用的参数"""
    parser.add_argument('--title', type=str, help='图表标题')
    parser.add_argument('--xlabel', type=str, help='X轴标签')
    parser.add_argument('--ylabel', type=str, help='Y轴标签')
    parser.add_argument('--ylim', type=float, nargs=2, help='Y轴范围, 例如: --ylim 0 5000')
    parser.add_argument('--output', '-o', type=str, choices=['plot', 'svg', 'png', 'pdf'],
                        default='plot', help='输出格式')
    parser.add_argument('--save-path', type=str, default='./plots', help='保存路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-legend', dest='legend', action='store_false', help='不显示图例')
    parser.set_defaults(legend=True)


def run_hydra_command(plot_type: str, args_dict: dict):
    """
    运行Hydra命令执行绘图

    参数:
        plot_type: 绘图类型
        args_dict: 参数字典
    """
    # 导入必要的模块
    from plotter.main import main
    from hydra.core.config_store import ConfigStore
    from hydra import compose, initialize
    from omegaconf import OmegaConf, DictConfig

    # 准备要传递给Hydra的参数
    overrides = [f"plot={plot_type}"]

    # 处理需要特殊格式化的参数
    for key, value in args_dict.items():
        if key == 'experiment_dirs' and value:
            dirs_str = "[" + ",".join(value) + "]"
            overrides.append(f"plot.experiment_dirs={dirs_str}")
        elif key == 'experiment_dir' and value:
            overrides.append(f"plot.experiment_dir={value}")
        elif key == 'title' and value:
            overrides.append(f"plot.title={value}")
        elif key == 'xlabel' and value:
            overrides.append(f"plot.xlabel={value}")
        elif key == 'ylabel' and value:
            overrides.append(f"plot.ylabel={value}")
        elif key == 'ylim' and value:
            ylim_str = f"[{value[0]},{value[1]}]"
            overrides.append(f"plot.ylim={ylim_str}")
        elif key == 'output' and value:
            overrides.append(f"plot.render_mode={value}")
        elif key == 'save_path' and value:
            overrides.append(f"plot.save_path={value}")
        elif key == 'debug' and value:
            overrides.append(f"debug={value}")
        elif key == 'legend' and value is not None:
            overrides.append(f"plot.legend={str(value).lower()}")
        elif key == 'baseline' and value is not None:
            overrides.append(f"plot.baseline_value={value}")
        elif key == 'offline' and value is not None:
            overrides.append(f"plot.offline_value={value}")
        elif key == 'window_size' and value is not None:
            overrides.append(f"plot.window_size={value}")

    logger.debug(f"Hydra overrides: {overrides}")

    # 初始化Hydra
    with initialize(version_base=None, config_path="conf"):
        # 组合配置
        cfg = compose(config_name="config", overrides=overrides)

        # 运行主函数
        return main(cfg)


def experiment_comparison(args):
    """执行实验比较绘图"""
    args_dict = vars(args)
    return run_hydra_command('experiment_comparison', args_dict)


def rollout_evaluation(args):
    """执行rollout与evaluation对比绘图"""
    args_dict = vars(args)
    return run_hydra_command('rollout_evaluation', args_dict)


def exception_rate(args):
    """执行异常率绘图"""
    args_dict = vars(args)
    return run_hydra_command('exception_rate', args_dict)


def learning_curve(args):
    """执行学习曲线绘图"""
    args_dict = vars(args)
    return run_hydra_command('learning_curve', args_dict)


def ablation_study(args):
    """执行消融研究绘图"""
    args_dict = vars(args)
    return run_hydra_command('ablation_study', args_dict)


def custom_plot(args):
    """执行自定义绘图"""
    args_dict = vars(args)
    return run_hydra_command('custom', args_dict)


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='实验绘图系统命令行工具')
    subparsers = parser.add_subparsers(dest='command', help='绘图命令')

    # 实验比较绘图
    exp_comp_parser = subparsers.add_parser('experiment_comparison',
                                           aliases=['exp', 'compare'],
                                           help='比较多个实验的结果')
    exp_comp_parser.add_argument('--dirs', dest='experiment_dirs', nargs='+', required=True,
                              help='实验目录列表，支持通配符')
    exp_comp_parser.add_argument('--filename', type=str, default='evaluation.csv',
                              help='数据文件名')
    exp_comp_parser.add_argument('--baseline', type=float, help='基准线值')
    exp_comp_parser.add_argument('--offline', type=float, help='离线值（起始点）')
    setup_common_args(exp_comp_parser)
    exp_comp_parser.set_defaults(func=experiment_comparison)

    # Rollout与Evaluation对比
    rollout_parser = subparsers.add_parser('rollout_evaluation',
                                        aliases=['rollout', 'eval'],
                                        help='比较rollout和evaluation性能')
    rollout_parser.add_argument('--dir', dest='experiment_dir', required=True,
                             help='实验目录，支持通配符')
    setup_common_args(rollout_parser)
    rollout_parser.set_defaults(func=rollout_evaluation)

    # 异常率分析
    exception_parser = subparsers.add_parser('exception_rate',
                                          aliases=['exception'],
                                          help='分析异常率')
    exception_parser.add_argument('--dir', dest='experiment_dir', required=True,
                               help='实验目录，支持通配符')
    setup_common_args(exception_parser)
    exception_parser.set_defaults(func=exception_rate)

    # 学习曲线
    learning_parser = subparsers.add_parser('learning_curve',
                                         aliases=['learning', 'curve'],
                                         help='绘制带有移动平均的学习曲线')
    learning_parser.add_argument('--dirs', dest='experiment_dirs', nargs='+', required=True,
                              help='实验目录列表，支持通配符')
    learning_parser.add_argument('--window', dest='window_size', type=int, default=10,
                              help='移动平均窗口大小')
    setup_common_args(learning_parser)
    learning_parser.set_defaults(func=learning_curve)

    # 消融研究
    ablation_parser = subparsers.add_parser('ablation_study',
                                         aliases=['ablation'],
                                         help='绘制消融研究的柱状图')
    setup_common_args(ablation_parser)
    ablation_parser.set_defaults(func=ablation_study)

    # 自定义绘图
    custom_parser = subparsers.add_parser('custom',
                                       help='使用自定义配置绘图')
    setup_common_args(custom_parser)
    custom_parser.set_defaults(func=custom_plot)

    # 解析参数
    args = parser.parse_args()

    # 如果没有提供命令，显示帮助
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1

    # 执行对应的函数
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
