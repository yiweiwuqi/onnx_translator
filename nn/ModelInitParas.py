# import onnx
# from nn import Tensor
# from nn import onnx_dtype_mapping
# import numpy as np
# from onnx import shape_inference

# # ONNX数据类型到NumPy数据类型的映射
# onnx_np_dtype_mapping = {
#  "float32": np.float32
# }

# def get_tensor_dtype(tensor_name, model):
#     """
#     获取张量的数据类型
    
#     Args:
#         tensor_name: 张量名称
#         model: ONNX模型对象
        
#     Returns:
#         int: ONNX数据类型编码，如果未找到返回None
#     """
#     # 对模型进行形状推断
#     inferred_model = shape_inference.infer_shapes(model)
#     graph = inferred_model.graph
    
#     # 在图的输入中查找
#     for input_tensor in graph.input:
#         if input_tensor.name == tensor_name:
#             return input_tensor.type.tensor_type.elem_type
            
#     # 在图的输出中查找
#     for output_tensor in graph.output:
#         if output_tensor.name == tensor_name:
#             return output_tensor.type.tensor_type.elem_type
            
#     # 在图的value_info中查找
#     for value_info_tensor in graph.value_info:
#         if value_info_tensor.name == tensor_name:
#             return value_info_tensor.type.tensor_type.elem_type
            
#     return None

# def ONNXParasGen(file_path):
#     """
#     从ONNX模型文件生成初始参数张量
    
#     Args:
#         file_path: ONNX模型文件路径
        
#     Returns:
#         tuple: (输入列表, 张量列表)
#     """
#     inputs_list = []
#     tensor_list = []
    
#     # 加载ONNX模型
#     model = onnx.load(file_path, load_external_data=False)
#     graph = model.graph
    
#     # 遍历图的输入节点
#     for item in graph.input:
#         print("item: ", item.name)
#         inputs_list.append(item.name)
#         # 提取张量维度信息
#         dimensions = [dim.dim_value for dim in item.type.tensor_type.shape.dim]
#         print("initial tensor dtype:",get_tensor_dtype(item.name, model))
#         # 获取张量数据类型
#         elem_type = get_tensor_dtype(item.name, model)
#         dtype = onnx_dtype_mapping[elem_type]
        
#         # 根据数据类型创建随机张量
#         if "float" in dtype:
#             tensor = Tensor(*dimensions, dtype=dtype)
#             tensor.data = np.random.rand(*dimensions).astype(onnx_np_dtype_mapping[dtype])
#             tensor_list.append(tensor)
#         else:
#             tensor = Tensor(*dimensions, dtype=dtype)
#             tensor.data = np.random.randint(-10, 10, size=dimensions, dtype=onnx_np_dtype_mapping[dtype])
#             tensor_list.append(tensor)
            
#     return inputs_list, tensor_list
import onnx
from nn import Tensor
from nn import onnx_dtype_mapping
import numpy as np
from onnx import shape_inference

# ONNX数据类型到NumPy数据类型的映射
onnx_np_dtype_mapping = {
 "float32": np.float32,
 "float16": np.float16,
 "int64": np.int64,
 "int32": np.int32,
 "bool": np.bool_,
}

def get_tensor_dtype(tensor_name, model):
    """
    获取张量的数据类型
    """
    # 对模型进行形状推断
    try:
        inferred_model = shape_inference.infer_shapes(model)
        graph = inferred_model.graph
    except:
        graph = model.graph
    
    # 在图的输入中查找
    for input_tensor in graph.input:
        if input_tensor.name == tensor_name:
            return input_tensor.type.tensor_type.elem_type
            
    # 在图的输出中查找
    for output_tensor in graph.output:
        if output_tensor.name == tensor_name:
            return output_tensor.type.tensor_type.elem_type
            
    # 在图的value_info中查找
    for value_info_tensor in graph.value_info:
        if value_info_tensor.name == tensor_name:
            return value_info_tensor.type.tensor_type.elem_type
            
    return None

def ONNXParasGen(file_path):
    """
    从ONNX模型文件生成初始参数张量 (已修复权重冲突问题)
    
    Args:
        file_path: ONNX模型文件路径
        
    Returns:
        tuple: (输入列表, 张量列表)
    """
    inputs_list = []
    tensor_list = []
    
    # 加载ONNX模型
    model = onnx.load(file_path, load_external_data=False)
    graph = model.graph

    # [Fix] 1. 构建所有 Initializer (权重) 的名称集合
    # 这些是内部常量，不应被视为外部输入
    initializer_names = set()
    for init in graph.initializer:
        initializer_names.add(init.name)
    
    # 遍历图的输入节点
    for item in graph.input:
        # [Fix] 2. 如果输入名称存在于 Initializer 中，跳过它
        if item.name in initializer_names:
            continue

        print("item: ", item.name)
        inputs_list.append(item.name)
        
        # 提取张量维度信息
        dimensions = [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        # 处理动态维度 (dim_value 为 0 或 None 的情况)，将其设为 1 以便进行模拟推断
        dimensions = [d if (d is not None and d > 0) else 1 for d in dimensions]

        # print("initial tensor dtype:", get_tensor_dtype(item.name, model))
        # 获取张量数据类型
        elem_type = get_tensor_dtype(item.name, model)
        dtype = onnx_dtype_mapping.get(elem_type, "float32")
        
        # 根据数据类型创建随机张量
        if "float" in dtype:
            tensor = Tensor(*dimensions, dtype=dtype)
            # 确保使用对应的 numpy 类型生成数据
            np_dtype = onnx_np_dtype_mapping.get(dtype, np.float32)
            tensor.data = np.random.rand(*dimensions).astype(np_dtype)
            tensor_list.append(tensor)
        else:
            tensor = Tensor(*dimensions, dtype=dtype)
            np_dtype = onnx_np_dtype_mapping.get(dtype, np.int32)
            tensor.data = np.random.randint(0, 2, size=dimensions).astype(np_dtype)
            tensor_list.append(tensor)
            
    return inputs_list, tensor_list