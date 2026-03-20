"""
合并所有 JSON 数据文件为统一的 JSONL 格式
自动处理编码和格式问题
"""

import json
import os
import re
import glob
from pathlib import Path


def try_load_json(filepath):
    """尝试多种方式加载 JSON 文件"""
    # 尝试标准加载
    for encoding in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    # 标准加载失败，尝试修复常见问题
    print(f"  标准解析失败，尝试修复: {os.path.basename(filepath)}")
    for encoding in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"  [FAIL] cannot read encoding: {filepath}")
        return []

    # 尝试逐个对象提取
    results = []
    # 匹配 {"instruction": ..., "output": ...} 块
    pattern = r'\{\s*"instruction"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"output"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
    matches = re.finditer(pattern, content, re.DOTALL)
    for m in matches:
        try:
            instruction = m.group(1).replace('\\"', '"').replace('\\n', '\n')
            output = m.group(2).replace('\\"', '"').replace('\\n', '\n')
            results.append({"instruction": instruction, "output": output})
        except Exception:
            continue

    if results:
        return results

    # 尝试修复尾部逗号等问题
    try:
        content = re.sub(r',\s*]', ']', content)
        content = re.sub(r',\s*}', '}', content)
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except Exception:
        pass

    # 最后手段：按 instruction/output 块拆分提取
    print(f"  trying block extraction...")
    results = []
    # 找到所有 "instruction" 开头的块
    parts = re.split(r'\{\s*"instruction"', content)
    for part in parts[1:]:  # 跳过第一个空块
        part = '{"instruction"' + part
        # 找到这个对象的结尾（下一个 }, 后面跟 { 或 ]）
        # 尝试逐字符找到完整的 JSON 对象
        depth = 0
        end_pos = -1
        in_string = False
        escape_next = False
        for i, c in enumerate(part):
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break

        if end_pos > 0:
            obj_str = part[:end_pos]
            try:
                obj = json.loads(obj_str)
                results.append(obj)
            except json.JSONDecodeError:
                # 尝试修复未转义的引号：在 output 值中替换
                try:
                    # 提取 instruction 和 output 的原始文本
                    inst_match = re.search(r'"instruction"\s*:\s*"', obj_str)
                    if inst_match:
                        results.append(_extract_fields(obj_str))
                except Exception:
                    pass

    if results:
        print(f"  recovered {len(results)} items via block extraction")
        return results

    print(f"  [FAIL] all parse methods failed: {filepath}")
    return []


def _extract_fields(text):
    """从可能有格式问题的 JSON 文本中提取 instruction 和 output"""
    # 找 "instruction": "..." 和 "output": "..."
    # 使用贪婪匹配找两个字段的边界
    inst_start = text.index('"instruction"')
    inst_val_start = text.index('"', inst_start + len('"instruction"') + 1) + 1

    out_marker = '"output"'
    out_pos = text.rindex(out_marker)
    # instruction 值结束在 "output" 前面的 ", 处
    inst_val_end = text.rindex('"', inst_val_start, out_pos)

    out_val_start = text.index('"', out_pos + len(out_marker) + 1) + 1
    out_val_end = text.rindex('"')

    instruction = text[inst_val_start:inst_val_end]
    output = text[out_val_start:out_val_end]

    # 反转义
    for old, new in [('\\n', '\n'), ('\\t', '\t'), ('\\"', '"'), ('\\\\', '\\')]:
        instruction = instruction.replace(old, new)
        output = output.replace(old, new)

    return {"instruction": instruction, "output": output}


def validate_example(example):
    """验证单条数据的格式"""
    if not isinstance(example, dict):
        return False
    if "instruction" not in example or "output" not in example:
        return False
    if len(example["instruction"].strip()) < 10:
        return False
    if len(example["output"].strip()) < 50:
        return False
    return True


def merge_all(data_dir, output_path):
    """合并所有 JSON 文件"""
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    all_examples = []
    stats = {}

    for filepath in json_files:
        filename = os.path.basename(filepath)
        examples = try_load_json(filepath)
        valid = [e for e in examples if validate_example(e)]
        stats[filename] = {"total": len(examples), "valid": len(valid)}
        all_examples.extend(valid)
        print(f"  {filename}: {len(valid)}/{len(examples)} 条有效")

    # 去重（基于 instruction 前 100 字符）
    seen = set()
    unique = []
    for e in all_examples:
        key = e["instruction"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(e)

    print(f"\n合并前: {len(all_examples)} 条")
    print(f"去重后: {len(unique)} 条")

    # 写入 JSONL
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for e in unique:
            row = {"instruction": e["instruction"], "output": e["output"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"已保存至: {output_path}")
    return len(unique)


if __name__ == "__main__":
    data_dir = Path(__file__).parent
    output_path = data_dir / "raw" / "all_data.jsonl"

    print("=" * 50)
    print("合并所有训练数据")
    print("=" * 50 + "\n")

    count = merge_all(str(data_dir), str(output_path))

    print(f"\n{'='*50}")
    print(f"合并完成! 共 {count} 条有效数据")
    print(f"{'='*50}")
    print(f"\n下一步运行:")
    print(f"  python data/prepare_dataset.py process --input data/raw/all_data.jsonl --output data/processed")
