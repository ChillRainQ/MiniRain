import os
import shutil

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import project


def _unwrap(model):
    """取出 DDP / torch.compile 包装下的原始模型"""
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw, '_orig_mod', raw)


def _atomic_save(obj, path):
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
        shutil.copytree(src, target, dirs_exist_ok=True)
    else:
        shutil.copy2(src, os.path.join(dst_root, os.path.basename(src)))

def _exp_dir(name: str):
    return os.path.join(project.PROJECT_ROOT_DIR, project.TRAIN_OUTPUT_DIR, name)


def load_ckpt(name: str, filename: str):
    """
    从检查点恢复
    """
    path = os.path.join(_exp_dir(name), filename)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location='cpu')
    saved_ws = ckpt.get('world_size', 1)
    current_ws = dist.get_world_size() if dist.is_initialized() else 1
    if saved_ws != current_ws and 'step' in ckpt:
        ckpt['step'] = ckpt['step'] * saved_ws // current_ws
        print(f'[load_ckpt] GPU数量变化({saved_ws}→{current_ws}), step换算为{ckpt["step"]}')
    return ckpt

def save_exp(name: str):
    """
    保存实验代码
    """
    exp_dir = os.path.join(project.PROJECT_ROOT_DIR, project.TRAIN_OUTPUT_DIR, name)
    os.makedirs(exp_dir, exist_ok=True)
    for entry in project.MODEL_CODE_DIR_AND_FILE:
        _copy_entry(entry, exp_dir)
    return exp_dir

def save_ckpt(path: str, model=None, epoch: int = 0, step: int = 0,
              optimizer=None, scheduler=None, scaler=None, **kwargs):
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
    :param kwargs: 其它需要保存的对象, 含 state_dict 的会自动调用, 否则原样存入
    :return: 检查点文件路径
    """
    ckpt = {
        'epoch': epoch,
        'step': step,
        'world_size': dist.get_world_size() if dist.is_initialized() else 1,
    }
    if model is not None:
        ckpt['model'] = {k: v.half().cpu() for k, v in _unwrap(model).state_dict().items()}
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

def save_state_dict(model, path: str, dtype=torch.float16):
    """
    保存模型参数(仅权重)
    :param model: 训练中的模型(可能为 DDP / torch.compile 包装)
    :param path: 目标 .pth 文件路径, 由外界维护
    :param dtype: 保存精度, 默认 fp16(与项目既有约定一致)
    :return: 权重文件路径
    """
    state_dict = {k: v.to(dtype).cpu() for k, v in _unwrap(model).state_dict().items()}
    _atomic_save(state_dict, path)
    del state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path
