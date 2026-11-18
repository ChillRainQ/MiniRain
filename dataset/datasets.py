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
    """
    MiniRain pretraining BIN dataset.

    Supports:
      - Unified all-text BIN/IDX and manifest
      - Original document BIN/IDX
      - Sequence Packed BIN/IDX

    BIN: uint16 little-endian
    IDX: uint64 little-endian, N + 1 offsets

    Packed mode:
      - No per-record BOS/EOS requirement
      - Uses original manifest for tokenizer metadata
      - Validates packed manifest when available
    """

    def __init__(
        self,
        bin_path,
        idx_path,
        max_length=4096,
        pad_token_id=None,
        pad_to_max=True,
        stage="p1",
        packed=None,
    ):
        super().__init__()

        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)
        self.max_length = int(max_length)
        self.pad_to_max = bool(pad_to_max)
        self.stage = stage

        # Automatically detect packed files.
        # Can be explicitly overridden with packed=True/False.
        self.packed = (
            "packed" in self.bin_path.stem.lower()
            if packed is None
            else bool(packed)
        )

        if not self.bin_path.is_file():
            raise FileNotFoundError(
                f"BIN not found: {self.bin_path}"
            )

        if not self.idx_path.is_file():
            raise FileNotFoundError(
                f"IDX not found: {self.idx_path}"
            )

        if self.max_length < 2:
            raise ValueError(
                "max_length must be >= 2"
            )

        # Memory-mapped binary arrays.
        self.tokens = np.memmap(
            self.bin_path,
            dtype="<u2",
            mode="r",
        )

        self.offsets = np.memmap(
            self.idx_path,
            dtype="<u8",
            mode="r",
        )

        # ---------------------------------------
        # Validate BIN / IDX
        # ---------------------------------------

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

        if np.any(
            self.offsets[1:] < self.offsets[:-1]
        ):
            raise RuntimeError(
                "IDX offsets are not monotonically increasing"
            )

        self.num_records = len(self.offsets) - 1

        lengths = (
            self.offsets[1:] - self.offsets[:-1]
        )

        if np.any(lengths < 2):
            raise RuntimeError(
                "BIN contains sequences shorter than 2 tokens. "
                "Check the final packed block."
            )

        if np.any(lengths > self.max_length):
            raise RuntimeError(
                "BIN contains sequences exceeding max_length"
            )

        # ---------------------------------------
        # Load original manifest
        # ---------------------------------------

        manifest_path = (
            self.bin_path.parent / "manifest.json"
        )

        if manifest_path.is_file():
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            # Original document mode must match
            # the original manifest exactly.
            if not self.packed:
                stage_meta = (manifest["dataset"]
                              if manifest.get("layout") == "unified_text"
                              else manifest["stages"][self.stage])

                if (
                    int(stage_meta["records"])
                    != self.num_records
                    or
                    int(stage_meta["stored_tokens"])
                    != len(self.tokens)
                    or
                    self.max_length
                    > int(manifest["max_length"])
                ):
                    raise RuntimeError(
                        "Original manifest does not match "
                        "BIN/IDX or max_length"
                    )

            # Packed mode uses the same tokenizer,
            # but record counts are different.
            else:
                if (
                    self.max_length
                    > int(manifest["max_length"])
                ):
                    raise RuntimeError(
                        "Packed sequence exceeds "
                        "original manifest max_length"
                    )

            special = manifest["special_tokens"]

            self.pad_id = int(
                special["pad_token_id"]
            )
            self.bos_id = int(
                special["bos_token_id"]
            )
            self.eos_id = int(
                special["eos_token_id"]
            )
            self.vocab_size = int(
                manifest["vocab_size"]
            )

            if (
                pad_token_id is not None
                and int(pad_token_id)
                != self.pad_id
            ):
                raise ValueError(
                    "pad_token_id does not match manifest"
                )

        else:
            self.pad_id = (
                0 if pad_token_id is None
                else int(pad_token_id)
            )
            self.bos_id = 1
            self.eos_id = 2
            self.vocab_size = 16384

        self.pad_token_id = self.pad_id

        # ---------------------------------------
        # Validate packed manifest if present
        # ---------------------------------------

        if self.packed:
            candidates = [
                self.bin_path.parent
                / f"{self.stage}_packed_manifest.json",
                self.bin_path.with_suffix(
                    ".manifest.json"
                ),
            ]

            packed_manifest_path = next(
                (p for p in candidates if p.is_file()),
                None,
            )

            if packed_manifest_path is not None:
                packed_meta = json.loads(
                    packed_manifest_path.read_text(
                        encoding="utf-8"
                    )
                )

                if (
                    int(packed_meta["packed_blocks"])
                    != self.num_records
                    or
                    int(packed_meta["packed_tokens"])
                    != len(self.tokens)
                    or
                    int(packed_meta["sequence_length"])
                    != self.max_length
                ):
                    raise RuntimeError(
                        "Packed manifest does not match "
                        "BIN/IDX or max_length"
                    )

        # ---------------------------------------
        # Print dataset information
        # ---------------------------------------

        print(
            "[PretrainBinDataset]\n"
            f"  Mode       : "
            f"{'PACKED' if self.packed else 'DOCUMENT'}\n"
            f"  Stage      : {self.stage}\n"
            f"  BIN        : {self.bin_path}\n"
            f"  IDX        : {self.idx_path}\n"
            f"  Records    : {self.num_records:,}\n"
            f"  Tokens     : {len(self.tokens):,}\n"
            f"  Max length : {self.max_length:,}\n"
            f"  Min length : {int(lengths.min()):,}\n"
            f"  Max record : {int(lengths.max()):,}\n"
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

        if length < 2:
            raise RuntimeError(
                f"Invalid record length: "
                f"index={index}, length={length}"
            )

        if length > self.max_length:
            raise RuntimeError(
                f"Compiled sample exceeds max_length: "
                f"index={index}, "
                f"length={length}, "
                f"max_length={self.max_length}"
            )

        ids_np = np.asarray(
            self.tokens[start:end],
            dtype=np.int64,
        ).copy()

        ids = torch.from_numpy(ids_np)

        # Original documents must start with BOS
        # and end with EOS.
        #
        # Packed blocks can begin or end in the
        # middle of a document.
        if not self.packed:
            if (
                ids[0].item() != self.bos_id
                or
                ids[-1].item() != self.eos_id
            ):
                raise RuntimeError(
                    f"Bad BOS/EOS at record {index}"
                )

        if not self.pad_to_max:
            return ids

        input_ids = torch.full(
            (self.max_length,),
            self.pad_token_id,
            dtype=torch.long,
        )

        labels = torch.full(
            (self.max_length,),
            -100,
            dtype=torch.long,
        )

        input_ids[:length] = ids
        labels[:length] = ids

        return input_ids, labels

    def get_length(self, index):
        if index < 0:
            index += self.num_records

        if index < 0 or index >= self.num_records:
            raise IndexError(index)

        return int(
            self.offsets[index + 1]
            - self.offsets[index]
        )

    def close(self):
        for name in ("tokens", "offsets"):
            array = getattr(self, name, None)

            if array is None:
                continue

            mmap = getattr(array, "_mmap", None)

            if mmap is not None:
                mmap.close()

            setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
