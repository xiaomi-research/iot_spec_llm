import json
import pandas as pd
import re

def extract_device_ids_from_json(data_str: str) -> list:
    """
    从JSON字符串中提取设备ID

    支持多种格式：
    1. Python字典格式：{'device_id': 'x1'} 或 [{'actions': [...]}]
    2. 标准JSON格式：{"device_id": "x1"} 或 [{"actions": [...]}]
    3. 字符串格式的JSON：'{"device_id": "x1"}'

    参数:
    data_str: 包含设备ID的字符串

    返回:
    list: 设备ID列表
    """
    try:
        if not data_str or pd.isna(data_str):
            return []

        # 转换为字符串并去除首尾空格
        data_str = str(data_str).strip()

        if not data_str:
            return []

        def extract_from_data(data):
            """从解析后的数据中提取设备ID的辅助函数"""
            device_ids = []

            if isinstance(data, list):
                # 数组格式: [{'actions': [...]}] 或 [{'device_id': 'x1'}]
                for item in data:
                    if isinstance(item, dict):
                        if 'actions' in item and isinstance(item['actions'], list):
                            # 处理actions数组
                            for action in item['actions']:
                                if isinstance(action, dict) and 'device_id' in action:
                                    device_ids.append(str(action['device_id']))
                        elif 'device_id' in item:
                            # 直接包含device_id
                            device_ids.append(str(item['device_id']))

            elif isinstance(data, dict):
                # 字典格式: {'actions': [...]} 或 {'device_id': 'x1'}
                if 'actions' in data and isinstance(data['actions'], list):
                    for action in data['actions']:
                        if isinstance(action, dict) and 'device_id' in action:
                            device_ids.append(str(action['device_id']))
                elif 'device_id' in data:
                    device_ids.append(str(data['device_id']))

            # 去重并返回
            return list(set(device_ids))

        # 方法1: 使用ast.literal_eval处理Python字典格式
        try:
            import ast
            data = ast.literal_eval(data_str)
            result = extract_from_data(data)
            if result:
                return result
        except (SyntaxError, ValueError, TypeError):
            # ast失败，继续尝试其他方法
            pass

        # 方法2: 尝试标准JSON解析
        try:
            data = json.loads(data_str)
            result = extract_from_data(data)
            if result:
                return result
        except json.JSONDecodeError:
            # JSON失败，继续尝试
            pass

        # 方法3: 尝试处理外层有引号的情况
        # 例如: '{"actions": [...]}' 或 "[{'actions': [...]}]"
        if (data_str.startswith('"') and data_str.endswith('"')) or \
                (data_str.startswith("'") and data_str.endswith("'")):
            inner_str = data_str[1:-1]

            # 再次尝试JSON
            try:
                data = json.loads(inner_str)
                result = extract_from_data(data)
                if result:
                    return result
            except json.JSONDecodeError:
                # 尝试ast
                try:
                    import ast
                    data = ast.literal_eval(inner_str)
                    result = extract_from_data(data)
                    if result:
                        return result
                except:
                    pass

        # 所有方法都失败
        return []

    except Exception as e:
        print(f"提取设备ID失败: {e}, 数据: {data_str[:100]}...")
        return []


def extract_device_ids_from_text(text: str, from_first_duplicate: bool = True) -> list:
    """
    从文本中提取所有设备ID，并进行去重和排序

    参数:
    text: 包含设备ID的文本
    from_first_duplicate: 是否从第一个重复的设备开始取

    返回:
    list: 去重并排序后的设备ID列表
    """
    if not text:
        return []

    # 正则表达式匹配 x1, x2, ..., x10, x11 等格式
    pattern = re.compile(r'x\d+')
    all_matches = pattern.findall(text)

    if not all_matches:
        return []

    # 如果不需要从第一个重复开始，直接返回所有设备ID（去重排序）
    if not from_first_duplicate:
        # 直接去重并排序
        final_devices = list(set(all_matches))
        final_devices.sort(key=lambda x: int(x[1:]))
        return final_devices

    # 如果需要从第一个重复开始（默认逻辑）
    # 找到所有设备，处理重复情况
    # 一旦有重复的，就从重复的开始往后取
    seen = set()
    start_index = 0

    # 找到第一个重复出现的位置
    for i, device in enumerate(all_matches):
        if device in seen:
            start_index = i
            break
        seen.add(device)
    else:
        # 如果没有重复，从第一个开始
        start_index = 0

    # 从第一个重复的位置开始取所有设备
    result_devices = all_matches[start_index:]

    # 再去重并保持顺序
    final_devices = []
    seen_devices = set()
    for device in result_devices:
        if device not in seen_devices:
            final_devices.append(device)
            seen_devices.add(device)

    # 按数字部分排序（x后面的数字）
    final_devices.sort(key=lambda x: int(x[1:]))

    return final_devices


def extract_device(text: str) -> tuple:
    """
    从文本中提取设备ID，按照优先级顺序匹配不同的关键词

    参数:
    text: 包含推理过程的完整文本

    返回:
    tuple: (提取出的内容, 去重后的设备ID列表)
    """
    # 定义匹配关键词的优先级顺序
    KEYWORDS = [
        r'最终输出的设备',
        r'最终召回设备',
        r'最终选择ID为',
        r'最终需要操作的设备',
        r'最终召回',
        r'最终确定召回'
        r'最终设备',
        r'最终确定要操作的设备',
        r'召回的设备',
        r'最终确定ID为',
        r'确定召回',
    ]

    # 1. 按优先级尝试匹配关键词
    for keyword in KEYWORDS:
        keyword_match = re.search(keyword, text)
        if keyword_match:
            # 从匹配位置开始提取到think结束标记之前的内容
            start_pos = keyword_match.start()
            remaining_text = text[start_pos:]

            # 查找think块的结束标记
            think_end_match = re.search(r'</think>', remaining_text)
            if think_end_match:
                extracted = remaining_text[:think_end_match.start()].strip()
            else:
                extracted = remaining_text.strip()

            # 从提取的内容中查找设备ID - 有关键词时，提取所有设备ID
            device_ids = extract_device_ids_from_text(extracted, from_first_duplicate=False)
            if device_ids:
                return extracted, device_ids

    # 2. 如果没有匹配到任何关键词，提取整个think块内容
    think_match = re.search(r'(.*?)</think>', text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()
        # 没有关键词时，按原来逻辑从第一次重复开始
        device_ids = extract_device_ids_from_text(think_text, from_first_duplicate=True)
        return think_text, device_ids

    # 3. 如果连think块都没有找到，返回空
    return "", []


def judge_consistency_by_device(text: str, output='') -> tuple:
    """
    判断think和instruction中的设备ID是否一致

    返回:
    tuple: (think_instruction_match, think_output_match, instruction_output_match,
            think_ids, instruction_ids, output_ids, text_content)
    """
    try:
        # 提取think设备
        _, think_ids = extract_device(text)

        # 提取instruction中的设备ID
        instruction_match = re.search(r'<instruction>(.*?)</instruction>', text, re.DOTALL)
        if instruction_match:
            instruction_str = instruction_match.group(1).strip()
            instruction_ids = extract_device_ids_from_json(instruction_str)
        else:
            instruction_ids = []

        # 提取output中的设备ID
        output_ids = extract_device_ids_from_json(output) if output else []

        # 转换为集合进行比较
        think_set = set(think_ids)
        instruction_set = set(instruction_ids)
        output_set = set(output_ids)

        print("think_set", think_set)
        print("instruction_set", instruction_set)
        print("output_set", output_set)
        # 计算匹配结果
        # 修改：如果instruction没有提取出设备ID，默认这一条是对的
        if not instruction_ids:
            think_instruction_match = True  # 默认正确
            instruction_output_match = True  # 默认正确
            think_output_match = think_set == output_set
        else:
            think_instruction_match = think_set == instruction_set
            think_output_match = think_set == output_set
            instruction_output_match = instruction_set == output_set

        return (think_instruction_match, think_output_match, instruction_output_match,
                think_ids, instruction_ids, output_ids, text)

    except Exception as e:
        print(f"错误: {e}")
        return False, False, False, [], [], [], text


if __name__ == '__main__':
    pred = '1. **query意图判断**：query为“现在打开客厅门口主灯”，操作品类为灯，动作是打开，属于支持的设备操作，且不缺少属性值，进入下一步。\n2. **属性功能过滤**：query中未指定属性，跳过此步骤，进入下一步。\n3. **抽取query中的品类并进行设备召回**：query中的设备品类为灯，是支持的品类，从用户设备列表中召回所有灯品类和开关品类（开关拓展品类为灯）的设备，即x1、x2、x3、x4、x5，进入下一步。\n4. **设备名全匹配召回**：query“现在打开客厅门口主灯”完全包含设备x1、x2、x3、x4、x5的设备名“主灯”，全匹配成功，直接选中这些设备，进入最终召回设备步骤。\n5. **最终召回设备**：最终召回的设备x1、x2、x3、x4在“小爱的家”的客厅门口，x5在“大葱的小家”的客厅门口，不在同一个房间内，进入异常处理的设备多选。根据召回的设备和query操作意图“打开”，这些设备都支持开关操作，所以最终需要操作的设备为x1、x2、x3、x4、x5。\n</think>\n<instruction>\n[{"actions": [{"device_id": "x1", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x2", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x3", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x4", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}, {"device_id": "x5", "action": "operate", "params": [{"spec_id": "property.2.1", "value": "true"}]}], "exception": {"type": "select", "value": ["x1", "x2", "x3", "x4", "x5"]}}]\n</instruction>'

    print(judge_consistency_by_device(pred))