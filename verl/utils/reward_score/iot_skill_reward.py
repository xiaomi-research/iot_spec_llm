# Copyright(C) 2026 Xiaomi Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from typing import Dict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from verl.utils.reward_tools import check_think2action

global_err_log = {}
def think_format(solution_str, err_reason):
    """
    根据solution_str的格式给予不同的奖励分数
    评分规则:
    - 格式1: "........\n</think>\n<instruction>\n.........\n</instruction>" 加1分
      (要求think内容和instruction内容都不能只有空格)
    - 格式2: "\n</think>\n<instruction>\n.........\n</instruction>" 加0.75分
      (要求instruction内容不能只有空格)
    - 格式3: 包含指定标签但不符合前两种格式 加0.2分
    - 其他情况: 0分
    """

    reward=0.0
    # 模式1: 包含think内容和instruction内容的完整格式
    # 例如: "一些思考内容</think><instruction>指令内容</instruction>"
    pattern1 = r'^(.+?)\n</think>\n<instruction>\n(.+?)\n</instruction>$'
    # 模式2: 只有instruction内容，think部分为空
    # 例如: "</think><instruction>指令内容</instruction>"
    pattern2 = r'^\n</think>\n<instruction>\n(.+?)\n</instruction>$'
    match1 = re.search(pattern1, solution_str, re.DOTALL)
    match2 = re.search(pattern2, solution_str, re.DOTALL)
    if match1:
        think_content = match1.group(1).strip()
        instruction_content = match1.group(2).strip()
        if think_content and instruction_content:
            return 2.0  # 提高完整格式的奖励
        elif instruction_content:
            err_reason.append("格式错误，未包含<think> tag")
            return 1.0
    elif match2:
        instruction_content = match2.group(1).strip()
        if instruction_content:
            err_reason.append("格式错误，未输出思考过程")
            return 1.0
        else:
            err_reason.append("格式错误，未输出spec")
            return 0.3
    # 检查是否有标签但格式错误
    has_think_tag = '</think>' in solution_str
    has_instruction_tags = '<instruction>' in solution_str and '</instruction>' in solution_str
    
    if has_think_tag and has_instruction_tags:
        err_reason.append("格式错误，未输出思考过程、spec；但包含</think>、<instruction>信息")
        return 0.2  # 有标签但格式错误
    elif has_think_tag or has_instruction_tags:
        err_reason.append("格式错误，未输出思考过程、spec；但包含</think>、<instruction>信息中的一个")
        return 0.1  # 只有部分标签

    err_reason.append("格式错误，完全错误")
    return -2.0  # 完全没有标签，给予负奖励

def non_think_format(solution_str, err_reason):
    if re.search(r'</think>|<instruction>|</instruction>', solution_str):
        err_reason.append("格式错误，不包含</think>|<instruction>|</instruction>")
        return -2.0
    return 3.0

def calculate_similarity(text1, text2):
    """
    计算两个英文加数字文本的BLEU相似度分数
    
    参数:
    text1: 第一个文本字符串
    text2: 第二个文本字符串
    
    返回:
    float: BLEU相似度分数 (0-1之间)
    """
    # 将文本分割成单词列表
    # 对于英文加数字文本，按空格分割即可
    reference = text2.split()
    candidate = text1.split()
    
    # 使用平滑函数处理零匹配的情况
    smoothing_function = SmoothingFunction().method1
    
    # 计算BLEU分数
    # 使用1-gram到4-gram的权重，可以根据需要调整
    bleu_score = sentence_bleu(
        [reference], 
        candidate, 
        weights=(0, 0, 0.2, 0.8),  # 1-gram到4-gram的权重
        smoothing_function=smoothing_function
    )
    
    return bleu_score

def is_string_equal(a, b):
    return a.strip() == b.strip()

def get_instruction_or_input(input_str:str):
    pattern = r"<instruction>\n(.*)\n</instruction>"
    match = re.search(pattern, input_str, re.DOTALL)
    ouput_text = match.group(1) if match else input_str
    ouput_text = ouput_text.strip()
    return ouput_text

def validate_json_format(text):
    """
    验证文本是否是有效的JSON格式
    """
    try:
        # 尝试解析JSON
        parsed = json.loads(text)
        return True, parsed
    except json.JSONDecodeError as e:
        return False, str(e)

def format_count_check(solution_str, err_reason):
    """
    增强的格式检查，包含JSON验证
    """
    # 原有的标签数量检查
    think_count = solution_str.count('</think>')
    instruction_open_count = solution_str.count('<instruction>')
    instruction_close_count = solution_str.count('</instruction>')
    
    if not (think_count == 1 and instruction_open_count == 1 and instruction_close_count == 1):
        err_reason.append("格式错误，不符合<think>、<instruction> 格式标准")
        return -1.0
    
    return 1.0

def separate_field_comparison(sol_parsed, gt_parsed):
    """分别对比device_id、spec_id和value字段"""

    def extract_all_values(data):
        """提取所有device_id、spec_id和value的值"""
        device_ids = set()
        spec_ids = set()
        values = set()

        def _get_element(obj, id_key, container):
            try:
                if id_key in obj:
                    if isinstance(obj[id_key], list):
                        for e in obj[id_key]:
                            container.add(e)
                    else:
                        container.add(obj[id_key])
            except e:
                print(id_key + ' 推理错误')

        def _extract(obj):
            try:
                if isinstance(obj, dict):
                    _get_element(obj, 'device_id', device_ids)
                    _get_element(obj, 'spec_id', spec_ids)
                    _get_element(obj, 'value', values)
                    # 递归处理所有值
                    for value in obj.values():
                        _extract(value)
                elif isinstance(obj, list):
                    for item in obj:
                        _extract(item)
            except:
                print("生成数据，格式错误")

        _extract(data)
        return device_ids, spec_ids, values

    # 提取解决方案和标准答案中的值
    sol_device_ids, sol_spec_ids, sol_values = extract_all_values(sol_parsed)
    gt_device_ids, gt_spec_ids, gt_values = extract_all_values(gt_parsed)

    # 对比结果
    device_id_match = sol_device_ids == gt_device_ids
    spec_id_match = sol_spec_ids == gt_spec_ids
    value_match = sol_values == gt_values

    return {
        'device_id': {
            'match': device_id_match,
            # 'solution_values': sorted(list(sol_device_ids)),
            # 'ground_truth_values': sorted(list(gt_device_ids)),
            # 'solution_only': sorted(list(sol_device_ids - gt_device_ids)),
            # 'ground_truth_only': sorted(list(gt_device_ids - sol_device_ids))
        },
        'spec_id': {
            'match': spec_id_match,
            # 'solution_values': sorted(list(sol_spec_ids)),
            # 'ground_truth_values': sorted(list(gt_spec_ids)),
            # 'solution_only': sorted(list(sol_spec_ids - gt_spec_ids)),
            # 'ground_truth_only': sorted(list(gt_spec_ids - sol_spec_ids))
        },
        'value': {
            'match': value_match,
            # 'solution_values': sorted(list(sol_values)),
            # 'ground_truth_values': sorted(list(gt_values)),
            # 'solution_only': sorted(list(sol_values - gt_values)),
            # 'ground_truth_only': sorted(list(gt_values - sol_values))
        },
        'all_match': device_id_match and spec_id_match and value_match
    }

def enhanced_answer_correct(solution_str, ground_truth, err_reason):
    """
    增强的答案正确性检查，包含JSON结构验证
    """
    solution_clean = get_instruction_or_input(solution_str)
    ground_truth_clean = get_instruction_or_input(ground_truth)
    
    # 检查JSON结构是否相同
    sol_valid, sol_parsed = validate_json_format(solution_clean)
    gt_valid, gt_parsed = validate_json_format(ground_truth_clean)
    if sol_valid and gt_valid:
        # 如果都是有效JSON，比较解析后的结构
        # 首先检查字符串完全相等
        # 然后比交字符串是否相似
        # TODO:应该按业务来计算相拟，首先判断设备id是否正确，然后是其它属性推理是否正确
        comparison_result = separate_field_comparison(sol_parsed, gt_parsed)
        if sol_parsed == gt_parsed:
            return 1.5
        elif comparison_result['device_id']['match']:
            err_reason.append("内容错误, device_id正确，其他错误")
            return 1.0
        elif comparison_result['spec_id']['match'] and comparison_result['value']['match']:
            err_reason.append("内容错误, device_id错误，其他正确")
            return 0.3
        else:
            err_reason.append("内容错误, 内容全错")
            return 0.1
        # elif calculate_similarity(solution_clean, ground_truth_clean) > 0.8:
        #     return 0.1
    err_reason.append("格式错误, JSON格式不对")
    return -1.0

def get_symmetry_bracket_reward(text:str, err_reason):
    """
    检查括号平衡性，专门检测缺少 [ 的问题
    """
    brackets = []
    for char in text:
        if char == '[' or char == '{':
            brackets.append(char)
        elif char == ']':
            if brackets and brackets[-1] == '[':
                brackets.pop()
            else:
                brackets.append(']')  # 多余的]
        elif char == '}':
            if brackets and brackets[-1] == '{':
                brackets.pop()
            else:
                brackets.append('}')  # 多余的}
    if len(brackets) == 0:
        return 0.5  # 括号平衡奖励
    else:
        err_reason.append("格式错误, 缺少括号")
        return -0.5  # 括号不平衡惩罚

def enhanced_think_rewards(solution_str, ground_truth, err_reason):
    """
    增强的思考模式奖励函数
    """
    reward = 0
    format_reward = think_format(solution_str, err_reason)
    reward += format_reward

    # if reward >= 1.0:
    #     format_check_reward = format_count_check(solution_str, err_reason)
    #     reward += format_check_reward
    #
    #     # 新增：括号对称检查
    #     instruction_content = get_instruction_or_input(solution_str)
    #     bracket_reward = get_symmetry_bracket_reward(instruction_content, err_reason)
    #     reward += bracket_reward

    correctness_reward = enhanced_answer_correct(solution_str, ground_truth, err_reason)
    reward += correctness_reward

    if check_think2action.judge_consistency_by_device(solution_str)[0]:
        reward += 0.5
    else:
        print("though-action 不一致", solution_str)

    return reward

def enhanced_non_think_rewards(solution_str, ground_truth, err_reason):
    """
    增强的非思考模式奖励函数
    """
    reward = 0
    reward += non_think_format(solution_str, err_reason)
    
    # 新增：括号平衡检查
    bracket_reward = get_symmetry_bracket_reward(solution_str, err_reason)
    reward += bracket_reward
    
    reward += enhanced_answer_correct(solution_str, ground_truth, err_reason)
    return reward

def compute_score(solution_str:str, ground_truth:str, extra_info:Dict[str, str]):
    """增强的评分函数"""
    if not solution_str or not ground_truth:
        return 0

    err_reason = []

    # 添加调试信息
    print(f"原始输出: {repr(solution_str)}")
    print(f"提取内容: {repr(get_instruction_or_input(solution_str))}")
    
    if "think" in extra_info and extra_info["think"]:
        reward = enhanced_think_rewards(solution_str, ground_truth, err_reason)
    else:
        reward = enhanced_non_think_rewards(solution_str, ground_truth, err_reason)
    
    print(f"最终奖励: {reward}  " + "; ".join(err_reason))
    for err in err_reason:
        global_err_log[err] = global_err_log.get(err, 0) + 1
    print(global_err_log)
    return reward

if __name__ == '__main__':
    pred = '<think>\n1. **query意图判断**：query为“现在打开客厅门口主灯”，操作品类为灯，动作是打开，属于支持的设备操作，且不缺少属性值，进入下一步。\n2. **属性功能过滤**：query中未指定属性，跳过此步骤，进入下一步。\n3. **抽取query中的品类并进行设备召回**：query中的设备品类为灯，是支持的品类，从用户设备列表中召回所有灯品类和开关品类（开关拓展品类为灯）的设备，即x1、x2、x3、x4、x5，进入下一步。\n4. **设备名全匹配召回**：query“现在打开客厅门口主灯”完全包含设备x1、x2、x3、x4、x5的设备名“主灯”，全匹配成功，直接选中这些设备，进入最终召回设备步骤。\n5. **最终召回设备**：最终召回的设备x1、x2、x3、x4在“小爱的家”的客厅门口，x5在“大葱的小家”的客厅门口，不在同一个房间内，进入异常处理的设备多选。根据召回的设备和query操作意图“打开”，这些设备都支持开关操作，所以最终需要操作的设备为x1、x2、x3、x4、x5。\n</think>\n<instruction>\n[{"actions": [{"device_id": "x1", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x2", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x3", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x4", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x5", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}], "exception": {"type": "select", "value": ["x1", "x2", "x3", "x4", "x5"]}}]\n</instruction>'
    ground_truth = '<instruction>\n[{"actions": [{"device_id": "x1", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x2", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x3", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x4", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x5", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}], "exception": {"type": "select", "value": ["x1", "x2", "x3", "x4", "x5"]}}]\n</instruction>'
    compute_score(pred, ground_truth, {"think": "think"})