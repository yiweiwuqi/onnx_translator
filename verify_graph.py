import nn
from nn import Graph
import nn.ModelInitParas
from nn.ONNXImport import ONNXImport
from nn.GraphVisualization import GraphGenerate
import os
import shutil

onnx_file_path = "./onnx_model/model.onnx"
task_name = "nps_verification"

result_dir = os.path.join("./result", task_name)
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)
print(f"📁 创建结果目录: {result_dir}")

def run_verification():
    print(f"\n🚀 开始验证模型: {onnx_file_path}")

    print("\n[Step 1] 正在运行 ONNXImport 导入算子...")
    try:
        ops_list = ONNXImport(onnx_file_path)
    except Exception as e:
        print(f"❌ 导入失败! 在解析 ONNX 节点时发生错误: {e}")
        return

    if not ops_list:
        print("❌ 错误: 未导入任何算子。请检查 ONNX 文件或导入器逻辑。")
        return
    
    print(f"✅ 成功导入 {len(ops_list)} 个算子节点。")
    print(f"   预览前 5 个算子: {[op.__class__.__name__ for op in ops_list[:5]]} ...")

    print("\n[Step 2] 解析模型初始输入参数...")
    try:
        initial_inputs, initial_tensors = nn.ModelInitParas.ONNXParasGen(onnx_file_path)
        print(f"   模型输入名称: {initial_inputs}")
        print(f"   输入张量形状: {[t.size for t in initial_tensors]}")
    except Exception as e:
        print(f"❌ 解析输入参数时出错: {e}")
        return

    print("\n[Step 3] 构建计算图 (Graph 对象)...")
    try:
        # 实例化 Graph 对象
        graph = Graph(
            ops=ops_list, 
            input_name=initial_inputs,
            model_name=task_name
        )
        
        print("   正在执行图结构推断 (forward_)...")
        placeholder_tensors = [nn.Tensor_(*t.size, dtype=t.dtype) for t in initial_tensors]
        graph.forward_(*placeholder_tensors)
        print("✅ 图结构推断完成，节点连接正常。")
        
    except Exception as e:
        print(f"❌ 构建图或推断形状时出错: {e}")
        
    print("\n[Step 4] 生成可视化流程图...")
    try:
        GraphGenerate(graph, task_name)
        print(f"✅ 可视化文件生成成功！请查看: {result_dir}/{task_name}.pdf")
        
        op_types = set([op.__class__.__name__ for op in ops_list])
        print(f"\n🎉 验证结束！共覆盖了 {len(op_types)} 种不同的算子类型:")
        print(f"   {list(op_types)}")
        
    except Exception as e:
        print(f"❌ 生成可视化图表时出错: {e}")

if __name__ == "__main__":
    if not os.path.exists(onnx_file_path):
        print(f"❌ 找不到模型文件: {onnx_file_path}")
        print("请先生成模型。")
    else:
        run_verification()