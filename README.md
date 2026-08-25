# VGThink

Code for **VGThink: Entity–Evidence Binding for End-to-End
Grounded Multimodal Named Entity Recognition**.

VGThink treats grounded multimodal named entity recognition (GMNER) as one
structured generation task: identify text mentions, predict their entity types,
and assign each mention either an image region or `None`.

1. annotation-grounded chain-of-thought construction;
2. Entity-Conditioned Evidence Bridge (ECEB) with official answer-box supervision;
3. group-relative sequence refinement with task, fine-grained, consistency, and
   format rewards.

## Installation

Python 3.10 or newer and a CUDA-capable GPU are recommended. Install the PyTorch
build appropriate for your CUDA environment first, then install this package:

```bash
conda create -n vgthink python=3.10 -y
conda activate vgthink
pip install -e .
```

For development checks:

```bash
pip install -e ".[test]"
pytest -q
```

The default backbone is
[`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
Transformers downloads it on first use. To use an existing local copy, set
`MODEL_NAME` when launching a script:

```bash
MODEL_NAME=/path/to/Qwen2.5-VL-7B-Instruct bash scripts/train_stage1.sh
```

## Data

The Twitter-GMNER benchmark, images, annotations, teacher rationales, and
converted data are not distributed here. Obtain the resources under their original
terms from the [GMNER dataset repository](https://github.com/NUSTM/GMNER), then
use the public [E2E-GMNER data preparation
workflow](https://github.com/Finch-coder/E2E-GMNER) to create chat-style SFT
files. This repository begins from those files; it does not duplicate the raw
TXT/XML-to-SFT conversion code.

Arrange the inputs and images, for example, as follows:

```text
data/
├── images/
│   └── sample.jpg
└── sft/
    ├── train_sft.jsonl
    ├── dev_sft.jsonl
    └── test_sft.jsonl
```

Each JSONL record must contain exactly one image, a user message, and an
assistant answer. Image paths may be absolute or relative to `--image-root`.
The user content must contain exactly one line beginning with `Text:`, followed
by the source post. Every answer mention must occur in that post; repeated
surface forms require distinct occurrences in annotation order so ECEB can bind
each record to its source tokens. Boxes use original-image pixel coordinates. A
compact input record, expanded below for readability, looks like:

```json
{
  "id": "example-1",
  "messages": [
    {"role": "user", "content": "Text: Alice visited ACME."},
    {
      "role": "assistant",
      "content": "<think>Teacher rationale.</think><answer>Alice|PER|[10,20,40,80]; ACME|ORG|None</answer>"
    }
  ],
  "images": ["images/sample.jpg"]
}
```

Store one such JSON object per line in each JSONL file.

The existing teacher `<think>` rationale is preserved when present. The builder
then appends deterministic evidence records derived only from the official
`<answer>`: box annotations become center-point `<obj>` anchors, while `None`
annotations become `<ungroundable>` records. It also creates normalized official
box targets for Stage II.

Prepare all three splits with:

```bash
python scripts/prepare_vgthink_data.py \
  --input-dir data/sft \
  --output-dir data/vgthink \
  --image-root data
```

The generated files are:

```text
data/vgthink/
├── train.jsonl
├── dev.jsonl
└── test.jsonl
```

Conversion fails fast on missing images, malformed official answers, and
malformed, non-finite, unordered, or empty boxes. The training loader additionally
rejects records that do not contain exactly one image. Use `train.jsonl` for
optimization, `dev.jsonl` for checkpoint selection, and reserve `test.jsonl` for
the final evaluation.

## Output grammar

The model is trained and instructed to emit one grounded trace followed by one
structured answer:

```text
<think>
... <obj>Alice|[25,50]</obj> <ungroundable>ACME</ungroundable> ...
</think>
<answer>Alice|PER|[10,20,40,80]; ACME|ORG|None</answer>
```

Trace points and answer boxes are generated in processed-image pixels and mapped
back to the original image before scoring. ECEB supervision uses separate
normalized official-box metadata and is never supplied as an inference input.

## Example training pipeline

```bash
bash scripts/train_stage1.sh
bash scripts/train_stage2.sh
bash scripts/train_stage3.sh
```

The stages form the following chain:

| Stage | Starts from | Main objective | Default output |
| --- | --- | --- | --- |
| I | Qwen2.5-VL | grounded-CoT response NLL | `outputs/stage1/best` |
| II | Stage-I adapter | field-weighted NLL + ECEB set-mass loss | `outputs/stage2/best` |
| III | Stage-II adapter and ECEB | group-relative policy loss + SFT anchor | `outputs/stage3/best` |

Stage II loads the Stage-I LoRA adapter and creates ECEB. In this implementation,
Stage III requires a complete Stage-II checkpoint and keeps ECEB active during
rollout, optimization, and inference. Each training stage evaluates `dev.jsonl`
after an epoch and selects `best` using development-set GMNER F1. The test split
is not read by the training entry points.

The examples accept environment-variable overrides for local paths and devices:

```bash
DEVICE=cuda:1 OUTPUT_DIR=outputs/my_stage1 bash scripts/train_stage1.sh

ADAPTER_PATH=outputs/my_stage1/best \
OUTPUT_DIR=outputs/my_stage2 \
bash scripts/train_stage2.sh

ADAPTER_PATH=outputs/my_stage2/best \
OUTPUT_DIR=outputs/my_stage3 \
bash scripts/train_stage3.sh
```

Available overrides include `MODEL_NAME`, `TRAIN_JSONL`, `DEV_JSONL`,
`IMAGE_ROOT`, `ADAPTER_PATH`, `OUTPUT_DIR`, `DEVICE`, and `PYTHON_BIN`. For other
experimental settings, inspect the compact command-line interfaces:

```bash
python train.py --help
python train_grpo.py --help
```

The root of every stage output records its effective CLI arguments in
`run_config.json`; the nested `best/` checkpoint does not duplicate that file.
Stage-II and Stage-III checkpoints additionally contain
`bridge_config.json` and `entity_patch_interactor.safetensors` alongside the
LoRA adapter and processor files.

## Evaluation

After checkpoint selection, evaluate the fixed best checkpoint on the test split:

```bash
bash scripts/evaluate.sh
```

Path and device overrides work in the same way:

```bash
CHECKPOINT=outputs/my_stage3/best \
TEST_JSONL=data/vgthink/test.jsonl \
OUTPUT_DIR=outputs/my_stage3/eval \
bash scripts/evaluate.sh
```

The default tag is `eval`, producing:

```text
outputs/eval/
├── test_pred_eval.jsonl
└── metrics_eval.json
```

The prediction file retains generated text, restored original-image coordinates,
and gold records for auditing. The metrics file reports precision, recall, and F1
for:

- **GMNER**: mention + type + region/`None`;
- **MNER**: mention + type;
- **EEG**: mention + region/`None`.

Localized records use the configured IoU threshold; `None` decisions require an
exact support-state match. Evaluation uses greedy decoding. Optional ECEB
diagnostics are available through `python evaluate.py --help`.

## Repository layout

```text
VGThink/
├── configs/system_prompt.txt       Shared training and inference protocol
├── scripts/
│   ├── prepare_vgthink_data.py    Grounded-CoT and region-target builder
│   ├── train_stage1.sh
│   ├── train_stage2.sh
│   ├── train_stage3.sh
│   └── evaluate.sh
├── src/e2egmner/
│   ├── data/                     Dataset and Qwen2.5-VL collator
│   ├── modeling/                 ECEB-enabled Qwen2.5-VL wrapper
│   ├── training/                 Entity–patch interaction and set-mass loss
│   ├── rl/                       Verifiable Stage-III reward
│   ├── evaluation/               Parsing, geometry, and metrics
│   └── engine/                   Shared dev/test generation loop
├── tests/                         Focused paper-component tests
├── train.py                       Stage-I/II entry point
├── train_grpo.py                  Stage-III entry point
└── evaluate.py                    Standalone evaluation entry point
```

## Acknowledgements

This implementation follows the data conventions and top-level workflow of
[E2E-GMNER](https://github.com/Finch-coder/E2E-GMNER). We also thank the authors
of [GMNER / H-Index](https://github.com/NUSTM/GMNER) and
[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) for their public
resources.

## Third-party licenses

Dataset and model licenses remain governed by their respective owners.
