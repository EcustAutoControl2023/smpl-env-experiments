#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import logging
from typing import Optional
import traceback

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('plotter')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plotter.plot_manager import PlotManager


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> Optional[int]:
    """
    主入口函数，使用Hydra管理配置
    
    使用方法:
        # 使用默认配置
        python -m plotter.main
        
        # 指定绘图类型
        python -m plotter.main plot=experiment_comparison
        
        # 覆盖配置中的特定值
        python -m plotter.main plot=rollout_evaluation plot.experiment_dir=path/to/experiment
        
        # 使用多组实验目录
        python -m plotter.main plot=experiment_comparison "plot.experiment_dirs=[path1,path2,path3]"
    """
    try:
        # 打印当前配置（可选，用于调试）
        if cfg.get('debug', False):
            logger.info("当前配置:\n%s", OmegaConf.to_yaml(cfg))
        
        # 检查关键配置
        if 'plot' not in cfg:
            logger.error("缺少'plot'配置部分")
            return 1
            
        if 'plot_type' not in cfg.plot:
            logger.error("缺少'plot_type'配置")
            return 1
        
        logger.info(f"正在使用绘图类型: {cfg.plot.plot_type}")
        
        # 创建并执行绘图管理器
        plot_manager = PlotManager(cfg.plot)
        plot_manager.execute()
        
        logger.info("绘图完成")
        return 0
        
    except Exception as e:
        logger.error(f"绘图过程中发生错误: {str(e)}")
        if cfg.get('debug', False):
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)