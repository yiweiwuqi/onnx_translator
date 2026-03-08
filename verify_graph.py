import nn
from nn import Graph
import nn.ModelInitParas
from nn.ONNXImport import ONNXImport
from nn.GraphVisualization import GraphGenerate
import os
import shutil
import traceback

#onnx_file_path = "./onnx_model/vad_ori_infer.onnx"
#onnx_file_path = "./onnx_model/bevdet_new_1.onnx"
#onnx_file_path = "./onnx_model/bevformer_tiny_epoch_24.onnx"
#onnx_file_path = "./onnx_model/uniad_dummy.onnx"
#onnx_file_path = "./onnx_model/simplify_extract_img_feat.onnx"
onnx_file_path = "./onnx_model/simplify_revised_head.onnx"
task_name = "nps_verification"

result_dir = os.path.join("./result", task_name)
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)
print(f" 创建结果目录: {result_dir}")

def run_verification():
    print(f"\n 开始验证模型: {onnx_file_path}")

    print("\n[Step 1] 正在运行 ONNXImport 导入算子...")
    try:
        ops_list = ONNXImport(onnx_file_path)
    except Exception as e:
        print(" 导入严重失败! 无法继续。错误堆栈:")
        traceback.print_exc()
        return

    if not ops_list:
        print(" 错误: 未导入任何算子。")
        return
    
    # 统计算子类型
    op_types = {}
    for op in ops_list:
        name = op.__class__.__name__
        if name == 'GenericNode':
            name = f"Generic({op.op_type})"
        op_types[name] = op_types.get(name, 0) + 1
        
    print(f" 成功导入 {len(ops_list)} 个算子节点。")
    print(f"  算子统计: {op_types}")

    print("\n[Step 2] 解析模型初始输入参数...")
    initial_inputs = []
    initial_tensors = []
    try:
        initial_inputs, initial_tensors = nn.ModelInitParas.ONNXParasGen(onnx_file_path)
        print(f"   模型输入名称: {initial_inputs}")
    except Exception as e:
        print(f" 警告: 解析输入参数出错: {e}")
        print("   将尝试使用空输入继续构建图...")

    print("\n[Step 3] 构建计算图并尝试形状推断...")
    graph = None
    try:
        # 实例化 Graph 对象
        graph = Graph(
            ops=ops_list, 
            input_name=initial_inputs,
            model_name=task_name
        )
        
        print("   正在执行图结构推断 (forward_)...")
        if initial_tensors:
            placeholder_tensors = [nn.Tensor_(*t.size, dtype=t.dtype) for t in initial_tensors]
            graph.forward_(*placeholder_tensors)
            print(" 图结构推断完成，节点连接逻辑验证通过。")
        else:
            print(" 跳过 forward_ (无输入张量)")
            
    except Exception as e:
        print(f" [非致命错误] Graph 将继续构建{e}")
        # traceback.print_exc() # 调试时可开启

    if graph is None:
        print(" 无法构建 Graph 对象，无法绘图。")
        return

    print("\n[Step 4] 生成可视化流程图...")
    try:
        GraphGenerate(graph, task_name)
        
    except Exception as e:
        print(f" 生成可视化图表时出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists(onnx_file_path):
        print(f" 找不到模型文件: {onnx_file_path}")
    else:
        run_verification()