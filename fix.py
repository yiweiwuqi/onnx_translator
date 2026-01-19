import os

file_path = "nn/Operators.py"
print(f"正在修复 {file_path} 中缺失 self.parameters 赋值的问题...")

# 定义带 self.parameters 赋值的正确模板
# 注意：每个方法的最后都增加了 self.parameters = {"values": values

# 2. Split (切分)
split_new = """    def forward_(self, input, split=None):
        try:
            num = len(self.outputs)
            # 即使 input 是 (1,) 也要造出 num 个 tensor
            out_tensors = [Tensor_(*input.size, dtype=self.dtype) for _ in range(num)]
        except:
            # 兜底
            out_tensors = [Tensor_(1, dtype=self.dtype) for _ in range(len(self.outputs))]
            
        values = {"tensor": out_tensors, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values"""

# 执行替换的辅助函数
def replace_method_in_class(content, class_name_marker, method_name, new_code):
    start_class = content.find(class_name_marker)
    if start_class == -1: return content
    
    start_method = content.find(method_name, start_class)
    if start_method == -1: return content
    
    # 找到方法结束 (下一个 def 或 class)
    end_method = len(content)
    next_def = content.find("\n    def ", start_method + 5)
    next_class = content.find("\nclass ", start_method + 5)
    
    if next_def != -1 and next_def < end_method: end_method = next_def
    if next_class != -1 and next_class < end_method: end_method = next_class
    
    print(f"✅ 修复 {class_name_marker} 的 {method_name}")
    return content[:start_method] + new_code + "\n" + content[end_method:]

# 读取
with open(file_path, 'r') as f:
    content = f.read()

# 替换列表
replacements = [
    ("class Split", "def forward_", split_new),
]

# 执行批量修复
for cls_marker, method, code in replacements:
    content = replace_method_in_class(content, cls_marker, method, code)

# 写入
with open(file_path, 'w') as f:
    f.write(content)

print("\n🚀 修复完成！可视化数据已补全。请再次运行 verify_full_graph.py")