# IoT Spec LLM

This project is a customized version of [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning) for training large language models on IoT device skill tasks. The main modification is a custom reward function designed specifically for evaluating IoT device operation outputs.

## Overview

This project focuses on training LLMs to understand and generate IoT device control specifications. The model learns to convert natural language queries into structured device control instructions, including device identification, action specification, and parameter configuration.

## Key Features

### Custom Reward Function

The primary modification is the custom reward function located at `verl/utils/reward_score/iot_skill_reward.py`. This reward function evaluates model outputs based on:

1. **Format Validation**
   - **Think Mode**: Validates the presence and correctness of `<think>` and `<instruction>` tags
   - **Non-Think Mode**: Ensures the output does not contain thinking tags
   - Checks for proper JSON structure and bracket symmetry

2. **Content Correctness**
   - Validates JSON format of the instruction content
   - Compares device IDs, spec IDs, and values between predicted and ground truth outputs
   - Provides granular scoring based on partial matches (e.g., correct device_id but wrong spec_id)

3. **Consistency Checking**
   - Verifies consistency between thinking process and action output (device ID matching)
   - Ensures the reasoning process aligns with the final instruction

### Reward Scoring Details

The reward function supports two modes:

- **Think Mode** (`enhanced_think_rewards`): 
  - Format reward: up to 2.0 points for complete format with both thinking and instruction
  - Correctness reward: up to 1.5 points for exact match, 1.0 for correct device_id, 0.3 for correct spec_id/value
  - Consistency reward: +0.5 points for think-action consistency

- **Non-Think Mode** (`enhanced_non_think_rewards`):
  - Format reward: 3.0 points for correct format (no thinking tags)
  - Bracket symmetry check: +0.5/-0.5 points
  - Correctness reward: same as think mode

## Dataset

The project includes training data for various IoT device categories:

- Air Conditioner (空调)
- Air Purifier (空气净化器)
- Clothes Drier (晾衣架)
- Curtain (窗帘)
- Dehumidifier (除湿机)
- Fan (风扇)
- Fresh Air Ventilator (新风机)
- Humidifier (加湿器)
- TV (电视)
- Window Opener (开窗器)
- And many more...

Data files are located in the `data/` directory, organized by device category.

## Installation

This project is based on verl. Please refer to the [verl documentation](https://github.com/volcengine/verl) for base installation instructions.

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For CUDA support:
```bash
pip install -r requirements-cuda.txt
```

For NPU support:
```bash
pip install -r requirements-npu.txt
```

## Usage

The reward function is automatically integrated into verl's training pipeline. When using `data_source="skill"` in your configuration, the custom reward function will be used.

### Example Configuration

```yaml
data:
  reward_fn_key: "data_source"
  data_source: "skill"

reward:
  custom_reward_function:
    path: "verl/utils/reward_score/iot_skill_reward.py"
    name: "compute_score"
```

### Reward Function API

```python
from verl.utils.reward_score.iot_skill_reward import compute_score

# Think mode
reward = compute_score(
    solution_str="<think>...</think><instruction>...</instruction>",
    ground_truth="<instruction>...</instruction>",
    extra_info={"think": "think"}
)

# Non-think mode
reward = compute_score(
    solution_str="[{\"actions\": [...]}]",
    ground_truth="[{\"actions\": [...]}]",
    extra_info={}
)
```

## Project Structure

```
iot_spec_llm/
├── data/                    # Training data for various IoT devices
├── verl/                    # Modified verl framework
│   └── utils/
│       └── reward_score/
│           └── iot_skill_reward.py  # Custom reward function
├── recipe/                  # Training recipes (from verl)
└── requirements*.txt        # Dependencies
```

## Main Modifications

Compared to the original verl project, this repository includes:

1. **Custom Reward Function** (`verl/utils/reward_score/iot_skill_reward.py`):
   - Format validation for IoT device control outputs
   - Structured content comparison (device_id, spec_id, value)
   - Think-action consistency checking
   - Detailed error logging and reporting

2. **IoT Device Data**: Training datasets for various smart home devices

All other components remain largely unchanged from the original verl framework.

## License

### Code
> [!NOTE]
> This project includes code derived from the **verl** project, which is licensed under the Apache License, Version 2.0.

Copyright (C) 2026 Xiaomi Corporation.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### Data
> [!NOTE]
> All datasets provided in this project are fully synthetic, contain no real user data or personal information, and are intended for illustrative and demonstration purposes only.

Copyright (C) 2026 Xiaomi Corporation.

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) (the "License"); you may not use this data except in compliance with the License. You may obtain a copy of the License at:
https://creativecommons.org/licenses/by-nc-sa/4.0/

Unless required by applicable law or agreed to in writing, data distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Acknowledgments

This project is based on [verl](https://github.com/volcengine/verl) by Bytedance - Seed - MLSys. We thank the verl team for their excellent work on reinforcement learning for LLMs.

## Citation

If you use this project in your research, please cite our paper:

```bibtex
@article{micu,
  title={MiCU: End-to-End Smart Home Command Understanding with Large Language Model},
  author={Han, Haowei and Hu, Kexin and Cai, Weiwei and Zhang, Debiao and Qin, Bin and Wang, Yuxiang and Jiang, Jiawei and Yan, Xiao and Du, Bo},
  year={2026}
}
```