import os
import re
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import project


def _unwrap(model):
    """取出 DDP / torch.compile 包装下的原始模型"""
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw, '_orig_mod', raw)


def _atomic_save(obj, path):
    path = os.fspath(path)
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _copy_entry(src: str, dst_root):
    if not os.path.exists(src):
        return
    if os.path.isdir(src):
        target = os.path.join(dst_root, os.path.basename(src))
        shutil.copytree(src, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    else:
        shutil.copy2(src, os.path.join(dst_root, os.path.basename(src)))

def _exp_dir(name: str):
    return os.path.join(project.PROJECT_ROOT_DIR, project.TRAIN_OUTPUT_DIR, name)


def load_ckpt_path(path: str, adjust_step_for_world_size: bool = True):
    """按完整路径加载检查点；精确续训可关闭 world-size step 换算。"""
    path = os.fspath(path)
    if not os.path.exists(path):
        return None
    # 训练断点包含 Python/NumPy RNG 状态，不能使用 weights_only=True。
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    saved_ws = ckpt.get('world_size', 1)
    current_ws = dist.get_world_size() if dist.is_initialized() else 1
    if adjust_step_for_world_size and saved_ws != current_ws and 'step' in ckpt:
        ckpt['step'] = ckpt['step'] * saved_ws // current_ws
        print(f'[load_ckpt] GPU数量变化({saved_ws}→{current_ws}), step换算为{ckpt["step"]}')
    return ckpt


def load_ckpt(name: str, filename: str, adjust_step_for_world_size: bool = True):
    """
    从检查点恢复
    """
    return load_ckpt_path(os.path.join(_exp_dir(name), filename),
                          adjust_step_for_world_size=adjust_step_for_world_size)

def save_exp(name: str, output_dir=None):
    """
    保存实验代码一次；续训或增加阶段时保留原始快照。
    """
    exp_dir = Path(output_dir if output_dir is not None else _exp_dir(name)).resolve()
    if exp_dir == Path(project.PROJECT_ROOT_DIR).resolve():
        raise ValueError('Experiment directory cannot be the project root')
    for entry in project.MODEL_CODE_DIR_AND_FILE:
        source = Path(entry).resolve()
        if source.is_dir() and exp_dir.is_relative_to(source):
            raise ValueError('Experiment directory cannot be inside a code source directory')
    exp_dir.mkdir(parents=True, exist_ok=True)
    marker = exp_dir / '.code_snapshot_complete'
    if marker.is_file():
        return str(exp_dir)
    for entry in project.MODEL_CODE_DIR_AND_FILE:
        _copy_entry(entry, exp_dir)
    marker.write_text('Experiment code snapshot saved.\n', encoding='utf8')
    return str(exp_dir)

def save_ckpt(path: str, model=None, epoch: int = 0, step: int = 0,
              optimizer=None, scheduler=None, scaler=None, model_dtype=None, **kwargs):
    """
    保存训练检查点(用于断点续训)
    打包模型权重 / 优化器 / 调度器 / 精度缩放器等状态到 path
    :param path: 检查点文件路径(.pth), 由外界维护
    :param model: 训练中的模型(可能为 DDP / torch.compile 包装)
    :param epoch: 当前 epoch
    :param step: 当前全局 step
    :param optimizer: 优化器(含 state_dict)
    :param scheduler: 学习率调度器(含 state_dict)
    :param scaler: 精度缩放器(含 state_dict)
    :param model_dtype: 可选的模型保存精度；None 保留原精度，适合精确续训
    :param kwargs: 其它需要保存的对象, 含 state_dict 的会自动调用, 否则原样存入
    :return: 检查点文件路径
    """
    ckpt = {
        'epoch': epoch,
        'step': step,
        'world_size': dist.get_world_size() if dist.is_initialized() else 1,
    }
    if model is not None:
        ckpt['model'] = {
            k: (v.to(dtype=model_dtype, device='cpu')
                if model_dtype is not None and (v.is_floating_point() or v.is_complex())
                else v.cpu())
            for k, v in _unwrap(model).state_dict().items()
        }
    for name, obj in (('optimizer', optimizer), ('scheduler', scheduler), ('scaler', scaler)):
        if obj is not None and hasattr(obj, 'state_dict'):
            ckpt[name] = obj.state_dict()
    for key, value in kwargs.items():
        if value is not None:
            ckpt[key] = value.state_dict() if hasattr(value, 'state_dict') else value
    _atomic_save(ckpt, path)
    del ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path

def save_rolling_ckpt(output_dir, *, step, keep=5, **kwargs):
    """保留最近 keep 个编号断点，另原子更新 latest.pt。仅 rank 0 调用。"""
    if keep < 1 or step < 0:
        raise ValueError('keep must be positive and step must be nonnegative')
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    numbered = directory / f'step_{step:08d}.pt'
    save_ckpt(numbered, step=step, **kwargs)
    latest = directory / 'latest.pt'
    temporary = directory / 'latest.pt.tmp'
    # 完整写入副本后替换，不会因复制中断破坏上一个 latest。
    shutil.copyfile(numbered, temporary)
    os.replace(temporary, latest)
    candidates = []
    for path in directory.iterdir():
        match = re.fullmatch(r'step_([0-9]+)\.pt', path.name)
        if match and path.is_file() and not path.is_symlink():
            candidates.append((int(match.group(1)), path))
    for _, path in sorted(candidates, key=lambda entry: entry[0])[:-keep]:
        # 只删除当前实验目录下严格匹配编号格式的文件。
        if path.resolve().parent != directory:
            raise ValueError(f'Checkpoint outside experiment directory: {path}')
        path.unlink()
    return numbered


def save_state_dict(model, path: str, dtype=torch.float16):
    """
    保存模型参数(仅权重)
    :param model: 训练中的模型(可能为 DDP / torch.compile 包装)
    :param path: 目标 .pth 文件路径, 由外界维护
    :param dtype: 保存精度, 默认 fp16(与项目既有约定一致)
    :return: 权重文件路径
    """
    state_dict = {
        k: (v.to(dtype=dtype, device='cpu')
            if v.is_floating_point() or v.is_complex() else v.cpu())
        for k, v in _unwrap(model).state_dict().items()
    }
    _atomic_save(state_dict, path)
    del state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path
