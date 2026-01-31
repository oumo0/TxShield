# 导入主要功能
from .api.analyzer_api import analyze_contract
from .config.settings import CONFIG
from .utils.file_utils import load_system_input

# 方便导入的别名
__all__ = [
    'analyze_contract',
    'CONFIG', 
    'load_system_input',
    'evaluation',
    'analysis',
    'experiments',
    'utils'
]

# 创建快捷导入
from . import evaluation
from . import analysis
from . import experiments
from . import utils