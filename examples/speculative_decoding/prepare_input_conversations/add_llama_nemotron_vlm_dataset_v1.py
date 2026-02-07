# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add Llama-Nemotron-VLM-Dataset-v1 conversations to a conversation dataset (VLM format)."""

import argparse
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from utils import (
    dataset_splits_explanation,
    id_for_conversation,
    update_dataset_file_with_conversations,
)

# Available splits in the dataset
AVAILABLE_SPLITS = [
    "captioning_1",
    "captioning_2",
    "ocr_1",
    "ocr_2",
    "ocr_3",
    "ocr_4",
    "ocr_5",
    "ocr_6",
    "ocr_7",
    "ocr_8",
    "ocr_9",
    "ocr_10",
    "vqa_1",
    "vqa_2",
    "vqa_3",
    "vqa_4",
    "vqa_5",
    "vqa_6",
    "vqa_7",
    "vqa_8",
    "vqa_9",
]

DATASET_REPO = "nvidia/Llama-Nemotron-VLM-Dataset-v1"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load Llama-Nemotron-VLM-Dataset-v1 conversations in VLM format."
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="ocr_1",
        help=f"""Split of the Llama-Nemotron-VLM-Dataset-v1 to load. Default is 'ocr_1'.
        Available splits: {", ".join(AVAILABLE_SPLITS)}""",
    )

    parser.add_argument(
        "--output-split-name",
        type=str,
        default=None,
        help=dataset_splits_explanation("llama-nemotron-vlm-v1-<dataset_split>")
        + "\nIf not provided, defaults to 'llama-nemotron-vlm-v1-<dataset_split>'.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input_conversations/"),
        help="Path to save conversations and images. Default is 'input_conversations/'.",
    )

    return parser.parse_args()


def download_images(dataset_split: str, image_dir: Path) -> None:
    """Download images for the specified split using huggingface_hub."""
    import tarfile

    images_folder = f"{dataset_split}_images"
    target_dir = image_dir / images_folder

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Images already exist at {target_dir}, skipping download.")
        return

    print(f"Downloading images for {dataset_split}...")
    image_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(image_dir),
        allow_patterns=[f"{images_folder}/*"],
    )

    for tar_path in target_dir.glob("*.tar") if target_dir.exists() else []:
        print(f"Found tar archive: {tar_path}, extracting...")
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                tar.extractall(path=target_dir)
            print(f"Extracted {tar_path}")
            tar_path.unlink()
        except Exception as e:
            print(f"Error extracting {tar_path}: {e}")
    print(f"Downloaded images to {target_dir}")


def parse_content_for_vlm(text: str, image_filename: str | None) -> list[dict]:
    """
    Parse text content to VLM format, preserving original <image> placeholder in text.

    Returns a list of content parts:
    - {"type": "image", "image": "<filename>"} for each <image> placeholder
    - {"type": "text", "text": "<original_text>"} with <image> preserved
    """
    content_parts = []

    # Add image entries for each <image> placeholder
    num_images = len(re.findall(r"<image>", text, flags=re.IGNORECASE))
    if image_filename:
        content_parts += [{"type": "image", "image": image_filename} for _ in range(num_images)]

    # Add the original text with <image> preserved
    content_parts.append({"type": "text", "text": text.replace("<image>", "")})

    return content_parts


async def main(args: argparse.Namespace) -> None:
    if args.output_split_name is None:
        args.output_split_name = f"llama-nemotron-vlm-v1-{args.dataset_split}"

    # Image directory is alongside output directory
    image_dir = args.output_dir / "images"

    # Download images first
    download_images(args.dataset_split, image_dir)

    # Load dataset
    ds = load_dataset(
        DATASET_REPO,
        split=args.dataset_split,
        streaming=False,
        verification_mode="no_checks",
    )

    input_conversations = []
    for i in tqdm(range(len(ds)), desc=f"Loading split {args.dataset_split}", total=len(ds)):
        entry = ds[i]
        conversations = entry.get("conversations", [])
        entry_id = entry.get("id", "")
        image_filename = entry.get("image", None)

        if not conversations or not isinstance(conversations, list):
            continue

        processed_conversations = []
        for msg in conversations:
            role = msg.get("from", msg.get("role", "")).lower()
            if not role:
                continue
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"

            raw_content = msg.get("value", msg.get("content", msg.get("text", "")))
            raw_content = raw_content.strip() if isinstance(raw_content, str) else str(raw_content)
            if not raw_content:
                continue

            if "<image>" in raw_content.lower():
                content = parse_content_for_vlm(raw_content, image_filename)
            else:
                content = [{"type": "text", "text": raw_content}]

            if content:
                processed_conversations.append({"role": role, "content": content})

        if processed_conversations:
            prompt_id = f"llama-nemotron-vlm-v1-{args.dataset_split}-{i:06}"
            if entry_id:
                prompt_id = f"{prompt_id}_{entry_id}"
            prompt_id = f"{prompt_id}_" + id_for_conversation(processed_conversations)
            input_conversations.append(
                {"conversation_id": prompt_id, "conversations": processed_conversations}
            )

    print(f"Loaded {len(input_conversations)} conversations from split {args.dataset_split}.")
    update_dataset_file_with_conversations(
        input_conversations, args.output_dir, args.output_split_name
    )


if __name__ == "__main__":
    import asyncio

    args = parse_args()
    asyncio.run(main(args))
