from datasets import load_dataset, Features, Sequence, Value
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    随机添加system
    """
    # 工具调用不处理
    if any(conv.get("tools") for conv in conversations):
        return conversations
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是MiniRain，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是MiniRain，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are MiniRain, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are MiniRain, a small but useful language model."
    ]
    # 可添加的
    if conversations[0].get('role') != 'system' and random.random() < add_system_ratio:
        return [
            {
                'role': 'system',
                'content': random.choice(SYSTEM_PROMPTS)
            }
        ] + conversations
    return conversations
    
    
    
    
    
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels




class PretrainBinDataset(Dataset):

    def __init__(self, bin_path, idx_path, max_length=4096, pad_token_id=None,
                 pad_to_max=True, stage="p1"):
        super().__init__()
        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)
        self.max_length = int(max_length)
        self.pad_to_max = bool(pad_to_max)
        self.stage = stage

        if not self.bin_path.is_file():
            raise FileNotFoundError(
                f"BIN not found: {self.bin_path}"
            )

        if not self.idx_path.is_file():
            raise FileNotFoundError(
                f"IDX not found: {self.idx_path}"
            )

        if self.max_length <= 0:
            raise ValueError(
                "max_length must be positive"
            )
        self.tokens = np.memmap(self.bin_path, dtype="<u2", mode="r",)
        self.offsets = np.memmap(self.idx_path, dtype="<u8", mode="r",)

        if len(self.offsets) < 2:
            raise RuntimeError(
                "IDX must contain at least 2 offsets"
            )

        if int(self.offsets[0]) != 0:
            raise RuntimeError(
                f"IDX first offset must be 0, "
                f"got {self.offsets[0]}"
            )

        if int(self.offsets[-1]) != len(self.tokens):
            raise RuntimeError(
                "BIN / IDX size mismatch:\n"
                f"last offset = {int(self.offsets[-1]):,}\n"
                f"bin tokens  = {len(self.tokens):,}"
            )

        # 防止损坏 idx
        if np.any(
            self.offsets[1:] < self.offsets[:-1]
        ):
            raise RuntimeError(
                "IDX offsets are not monotonically increasing"
            )

        self.num_records = len(self.offsets) - 1
        lengths = self.offsets[1:] - self.offsets[:-1]
        if np.any(lengths < 2) or np.any(lengths > self.max_length):
            raise RuntimeError("BIN contains sequences outside [2, max_length]")

        manifest_path = self.bin_path.parent / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf8"))
            stage_meta = manifest["stages"][self.stage]
            if (int(stage_meta["records"]) != self.num_records or
                    int(stage_meta["stored_tokens"]) != len(self.tokens) or
                    self.max_length > int(manifest["max_length"])):
                raise RuntimeError("manifest does not match BIN/IDX or max_length")
            special = manifest["special_tokens"]
            manifest_pad = int(special["pad_token_id"])
            if pad_token_id is not None and int(pad_token_id) != manifest_pad:
                raise ValueError("pad_token_id does not match manifest")
            self.pad_id = manifest_pad
            self.bos_id = int(special["bos_token_id"])
            self.eos_id = int(special["eos_token_id"])
            self.vocab_size = int(manifest["vocab_size"])
        else:
            self.pad_id = 0 if pad_token_id is None else int(pad_token_id)
            self.bos_id, self.eos_id, self.vocab_size = 1, 2, 16384
        self.pad_token_id = self.pad_id

        print(
            "[PretrainBinDataset]\n"
            f"  BIN        : {self.bin_path}\n"
            f"  IDX        : {self.idx_path}\n"
            f"  Records    : {self.num_records:,}\n"
            f"  Tokens     : {len(self.tokens):,}\n"
            f"  Max length : {self.max_length:,}\n"
            f"  BIN size   : "
            f"{self.bin_path.stat().st_size / 1e9:.3f} GB"
        )

    def __len__(self):
        return self.num_records

    def __getitem__(self, index):
        if index < 0:
            index += self.num_records

        if index < 0 or index >= self.num_records:
            raise IndexError(index)

        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        length = end - start

        if length <= 0:
            raise RuntimeError(
                f"Empty record: index={index}"
            )
        if length > self.max_length:
            raise RuntimeError(
                f"Compiled sample exceeds max_length: "
                f"index={index}, "
                f"length={length}, "
                f"max_length={self.max_length}"
            )
        ids_np = np.asarray(self.tokens[start:end], dtype=np.int64,).copy()

        ids = torch.from_numpy(ids_np)
        if ids[0].item() != self.bos_id or ids[-1].item() != self.eos_id:
            raise RuntimeError(f"Bad BOS/EOS at record {index}")
        if not self.pad_to_max:
            return ids

        input_ids = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long,)
        labels = torch.full((self.max_length,), -100, dtype=torch.long,)
        input_ids[:length] = ids
        labels[:length] = ids
        return input_ids, labels

    def get_length(self, index):
        return int(self.offsets[index + 1] - self.offsets[index])

    def close(self):
        for name in ("tokens", "offsets"):
            array = getattr(self, name, None)
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
                array._mmap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_seq_len
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        # 加入对工具调用的支持
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message.get("tools")) if isinstance(message.get("tools"), str) else message.get("tools")
                
            if message.get("tool_calls") and isinstance(message.get("tool_calls"), str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
        
    def generate_labels(self, input_ids):
        # 初始化
        labels = [-100] * len(input_ids)
        # 生成时只给assistant的token赋值
        i = 0
        while i < len(input_ids):
            # 这里面是一个兼容性情况，BOS可能是多token组成的
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start 
                # 找EOS
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                    
                for j in range(start, min(end + len(self.eos_id), self.max_len)):
                    labels[j] = input_ids[j]
                    
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # 预处理
        conversations = pre_processing_chat(sample['conversations'])
        # prompt处理
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_len]
        # padding
        input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
        
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
                
    
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }
