import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2
from Memory.memory import two_level_ebbinghaus_indices

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# R2R/VLN-CE TSV：id 形如 "12/step_0003_TURN_RIGHT.png" 或 "..._STOP.png"
_VLN_STEP_ID_RE = re.compile(r"^([^/]+)/step_(\d+)_", re.IGNORECASE)


def build_chunk_meta_from_vln_annotation(raw: Dict) -> Optional[Dict]:
    """供 Scene Bank / MemBank：从标注推断 is_first_chunk、is_last_step、episode_id。"""
    sid = raw.get("id")
    if not isinstance(sid, str) or not sid.strip():
        return None
    m = _VLN_STEP_ID_RE.match(sid.strip())
    if not m:
        return None
    episode_id = m.group(1)
    step_idx = int(m.group(2))
    is_first_chunk = step_idx == 0
    sid_up = sid.upper()
    is_last_step = "_STOP" in sid_up or sid_up.endswith("_STOP.PNG")
    if not is_last_step:
        for c in reversed(raw.get("conversations") or []):
            if c.get("from") == "gpt":
                if (c.get("value") or "").strip() == "STOP":
                    is_last_step = True
                break
    return {
        "is_first_chunk": is_first_chunk,
        "is_last_step": is_last_step,
        "episode_id": episode_id,
    }


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path, max_samples: int=-1):
    with open(path, "r") as f:
        # return [json.loads(line) for line in f]
        ret = []
        for line in f:
            ret.append(json.loads(line))
            if max_samples !=-1 and len(ret) >= max_samples:
                break
    return ret


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"], max_samples=data_args.max_samples)
            elif file_format == "tsv":
                annotations = []
                with open(data["annotation_path"], "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        cols = line.split("\t")
                        parsed = None
                        for col in reversed(cols):
                            col = col.strip()
                            if not col:
                                continue
                            try:
                                parsed = json.loads(col)
                                break
                            except Exception:
                                continue
                        if parsed is None:
                            parsed = {"columns": cols}
                        annotations.append(parsed)
            else:
                annotations = json.load(open(data["annotation_path"], "r", encoding="utf-8"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
                ann["tag"] = data["tag"]
            list_data_dict += annotations

        print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                video_cap = int(getattr(self.data_args, "video_max_frames", -1) or -1)
                image_num = video_cap if video_cap > 0 else int(
                    getattr(self.data_args, "shrink_min_frames", 16)
                )
            else:
                image_num = 0
            length_list.append(image_num * 252 + cur_len)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                video_cap = int(getattr(self.data_args, "video_max_frames", -1) or -1)
                image_num = video_cap if video_cap > 0 else int(
                    getattr(self.data_args, "shrink_min_frames", 16)
                )
            else:
                image_num = 0
            cur_len += image_num*252
            tag = sample.get("tag", "2d")
            cur_len = -cur_len if tag == "2d" else cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw
    
    def draw_visual_marks(self, images, spar_info):
        # Images: list[PIL.Image.Image] when the number of images > 1
        # or PIL.Image.Image otherwise
        if spar_info is None:
            return
        info = json.loads(spar_info)
        task_type = info["type"]
        from .draw_marker import DRAW_FUNCTIONS
        draw_fn = DRAW_FUNCTIONS[task_type]
        draw_fn(images, info)
        # for j, img in enumerate(images):
        #     # write to local
        #     img.save(f"images/img_{j}.jpg", format="JPEG")

    def process_image_unified_vggt(self, image_file):
        # this function reshapes the image to the width size of 518
        image_processor = copy.deepcopy(self.data_args.image_processor)
        from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
        images = load_and_preprocess_images([image_file])
        images_vggt = copy.deepcopy(images[0])
        merge_size: int = getattr(image_processor, "merge_size")
        patch_size: int = getattr(image_processor, "patch_size")
        _, height, width = images[0].shape
        # merge_size = 2
        if (width // patch_size) % merge_size > 0:
            width = width - (width // patch_size) % merge_size * patch_size
        if (height // patch_size) % merge_size > 0:
            height = height - (height // patch_size) % merge_size * patch_size
        images = images[:,:, :height, :width]
        visual_processed = image_processor(images, return_tensors="pt", do_rescale=False)
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        # return image_tensor, grid_thw
        return {
            "pixel_values": image_tensor,
            "image_grid_thw": grid_thw,
            "images_vggt": images_vggt
        }

    def _sample_video_frame_indices(self, num_frames: int) -> List[int]:
        """Apply the MemoryVLN adaptive two-level Ebbinghaus policy."""
        if num_frames <= 0:
            return []
        if getattr(self.data_args, "skip_frame_sampling", False):
            return list(range(num_frames))

        start_frames = int(getattr(self.data_args, "shrink_start_frames", 16))
        if num_frames <= start_frames:
            return list(range(num_frames))

        adaptive_offset = float(getattr(self.data_args, "shrink_adaptive_offset", 4.0))
        adaptive_coef = float(getattr(self.data_args, "shrink_adaptive_coef", 2.0))
        min_frames = int(getattr(self.data_args, "shrink_min_frames", 16))
        target_frames = max(
            min(num_frames, int(adaptive_offset + adaptive_coef * (num_frames ** 0.5))),
            min_frames,
        )

        hard_cap = int(getattr(self.data_args, "video_max_frames", -1) or -1)
        if hard_cap > 0:
            target_frames = min(target_frames, hard_cap)
        if num_frames <= target_frames:
            return list(range(num_frames))

        return two_level_ebbinghaus_indices(
            num_frames=num_frames,
            max_frames=target_frames,
            chunk_size=int(getattr(self.data_args, "ebbinghaus_chunk_size", 8)),
            s_inner=float(getattr(self.data_args, "ebbinghaus_s", 5.0)),
            s_outer=float(getattr(self.data_args, "ebbinghaus_s_outer", 3.0)),
        )

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps

        frame_idx = np.array(self._sample_video_frame_indices(total_frames), dtype=np.int64)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts
    

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e
    
    def read_video_images(self, source):
        # read video images from the source
        assert isinstance(source["video"], str), "video should be a string"
        video_file = os.path.join(source["data_path"], source["video"])
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
            raise FileNotFoundError

        # check whether video_file is a directory
        if os.path.isdir(video_file):
            frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
            frame_files.sort()
            frame_idx = self._sample_video_frame_indices(len(frame_files))
            images = [frame_files[i] for i in frame_idx]
            images = [Image.open(frame).convert("RGB") for frame in images]
        elif any([video_file.endswith(ext) for ext in [".mp4", ".avi", ".mov"]]):
            vr = VideoReader(video_file, num_threads=4)
            total_frames = len(vr)
            avg_fps = vr.get_avg_fps()
            frame_idx = self._sample_video_frame_indices(total_frames)
            video = vr.get_batch(frame_idx).asnumpy()
            
            images = [Image.fromarray(frame).convert("RGB") for frame in video]
        return images

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        video = None
        
        if "video" in sources[0]:
            sources[0]["images"] = self.read_video_images(sources[0])
            num_image = len(sources[0]["images"])
            sources[0]["conversations"][0]["value"] = sources[0]["conversations"][0]["value"].replace(
                DEFAULT_VIDEO_TOKEN, "".join([DEFAULT_IMAGE_TOKEN] * num_image)
            )
            del sources[0]["video"]
        
        # # replace <image>\n with <image>
        sources[0]["conversations"][0]["value"] = sources[0]["conversations"][0]["value"].replace(
            f"{DEFAULT_IMAGE_TOKEN}\n", DEFAULT_IMAGE_TOKEN
        )

        # rename images tag
        if "images" in sources[0]:
            sources[0]["image"] = sources[0]["images"]

        # notice that we use images as the tag（不全局抽帧：与标注中的图像/位姿一一对应）
        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    if isinstance(image_file[0], str):
                        image_file = [
                            os.path.join(image_folder, file) for file in image_file
                        ]
                    elif isinstance(image_file[0], Image.Image):
                        pass
                    else:
                        raise NotImplementedError

                    if isinstance(image_file[0], str):
                        image_file = [Image.open(img).convert("RGB") for img in image_file]

                    self.draw_visual_marks(image_file, sources[0].get("spar_info", None))

                    image, grid_thw, images_vggt = [], [], []

                    for file in image_file:
                        ret = self.process_image_unified_vggt(file)
                        image.append(ret["pixel_values"])
                        images_vggt.append(ret["images_vggt"])
                        grid_thw.append(ret["image_grid_thw"])
                else:
                    image_file = image_file[0]
                    if isinstance(image_file, str):
                        image_file = os.path.join(image_folder, image_file)
                        image_file = Image.open(image_file).convert("RGB")
                    elif isinstance(image_file, Image.Image):
                        pass
                    else:
                        raise NotImplementedError
                    self.draw_visual_marks(image_file, sources[0].get("spar_info", None))
                    ret = self.process_image_unified_vggt(image_file)
                    image = [ret["pixel_values"]]
                    images_vggt = [ret["images_vggt"]]
                    grid_thw = ret["image_grid_thw"]
            else:
                raise NotImplementedError

            if not isinstance(grid_thw, list):
                grid_thw = [grid_thw]
            K = len(grid_thw)
            text = sources[0]["conversations"][0]["value"]
            n_image_tokens_in_text = text.count(DEFAULT_IMAGE_TOKEN)
            if K > 1 and n_image_tokens_in_text != K:
                suffix_start = text.rfind(DEFAULT_IMAGE_TOKEN) + len(DEFAULT_IMAGE_TOKEN)
                suffix_text = text[suffix_start:]
                new_text = (
                    "You are a visual language navigation model, and your should go to the "
                    "locations to complete the given task. Compare the observation and "
                    "instruction to infer your current progress, and then select the correct "
                    "direction from the candidates to go to the target location and finish "
                    "the task.\n"
                    " Your observation memories:"
                    + DEFAULT_IMAGE_TOKEN * (K - 1)
                    + "\n Your current observation:"
                    + DEFAULT_IMAGE_TOKEN
                    + suffix_text
                )
                sources[0]["conversations"][0]["value"] = new_text

            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image"
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )
        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video"
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
            )

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
            data_dict["images_vggt"] = images_vggt
            poses_raw = self.list_data_dict[i].get("agent_poses")
            num_imgs = len(grid_thw) if isinstance(grid_thw, list) else 1
            if poses_raw is not None and len(poses_raw) == num_imgs:
                poses_6d = []
                for p in poses_raw:
                    if isinstance(p, dict):
                        pos = p.get("position", [0, 0, 0])
                        euler = p.get("rotation_euler", [0, 0, 0])
                        poses_6d.append(
                            [
                                float(pos[0]),
                                float(pos[1]),
                                float(pos[2]),
                                float(euler[0]),
                                float(euler[1]),
                                float(euler[2]),
                            ]
                        )
                    else:
                        poses_6d.append(p)
                data_dict["agent_poses"] = torch.tensor(poses_6d, dtype=torch.float32)
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw
        
        data_dict["tag"] = self.list_data_dict[i].get("tag", "2d")
        cm = build_chunk_meta_from_vln_annotation(self.list_data_dict[i])
        if cm is not None:
            data_dict["chunk_meta"] = cm
        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
                
        # assume all data in a batch has images_vggt
        if "images_vggt" in instances[0]:
            images_vggt = [torch.stack(instance["images_vggt"]) for instance in instances]
            batch["images_vggt"] = images_vggt
            assert len(set([instance["tag"] for instance in instances])) == 1, "all data in a batch should have the same tag"
            batch["tag"] = instances[0]["tag"]
        if all(inst.get("agent_poses") is not None for inst in instances):
            batch["agent_poses"] = torch.cat(
                [inst["agent_poses"] for inst in instances], dim=0
            )
        if any("chunk_meta" in inst for inst in instances):
            metas = [inst.get("chunk_meta") for inst in instances]
            batch["chunk_meta"] = metas[0] if len(metas) == 1 else metas
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )

        seq_lens = torch.tensor(
            [0] + [len(seq) for seq in input_ids], dtype=torch.int32
        )
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

                
        # assume all data in a batch has images_vggt
        if "images_vggt" in instances[0]:
            raise NotImplementedError("FlattenedDataCollatorForSupervisedDataset does not support images_vggt")

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
