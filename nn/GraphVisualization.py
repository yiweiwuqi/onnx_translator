from graphviz import Digraph
import os

# def GraphGenerate(input_graph, model_name):
#     """
#     生成计算图的可视化图形
    
#     Args:
#         input_graph: 输入的计算图对象，包含操作节点和连接关系
#         model_name: 模型名称，用于保存图形文件
        
#     Returns:
#         None: 直接生成并保存图形文件
#     """
#     # 创建有向图对象
#     dot = Digraph()
    
#     # 提取所有操作节点名称
#     nodes_data = []
#     for item in input_graph.ops.keys():
#         nodes_data.append(item)
    
#     # 收集所有输入和输出节点名称
#     temp_nodes = []
#     for item in nodes_data:
#         for i in input_graph.ops[item].inputs:
#             temp_nodes.append(str(i))
#         for i in input_graph.ops[item].outputs:
#             temp_nodes.append(str(i))
    
#     # 去重处理
#     temp_nodes = list(set(temp_nodes))

#     # 为每个操作节点创建图形节点
#     for node_data in nodes_data:
#         node_name = node_data
#         # 根据输出张量的数量和维度信息创建不同的标签
#         if isinstance(input_graph.ops[node_name].parameters["values"]["tensor"], list):
#             if len(input_graph.ops[node_name].parameters["values"]["tensor"]) == 2:
#                 label = (f'{{ input node: {input_graph.ops[node_name].inputs} | {node_name}| '
#                          f'output node: {input_graph.ops[node_name].outputs}| '
#                          f'output tensor: {[input_graph.ops[node_name].parameters["values"]["tensor"][0].size,input_graph.ops[node_name].parameters["values"]["tensor"][1].size]} }}')
#                 dot.node(node_name, label=label, shape='record', width='1.6', height='0.8',
#                          fontsize="12",
#                          style='rounded,filled', 
#                          fillcolor='lightblue', color='blue', fontname='Consolas')

#             elif len(input_graph.ops[node_name].parameters["values"]["tensor"]) == 3:
#                 label = (f'{{ input node: {input_graph.ops[node_name].inputs} | {node_name}| '
#                          f'output node: {input_graph.ops[node_name].outputs}| '
#                          f'output tensor: {[input_graph.ops[node_name].parameters["values"]["tensor"][0].size, input_graph.ops[node_name].parameters["values"]["tensor"][1].size, input_graph.ops[node_name].parameters["values"]["tensor"][2].size]} }}')
#                 dot.node(node_name, label=label, shape='record', width='1.6', height='0.8',
#                          fontsize="12",
#                          style='rounded,filled', 
#                          fillcolor='lightblue', color='blue', fontname='Consolas')

#             else:
#                 label = (f'{{ input node: {input_graph.ops[node_name].inputs} | {node_name}| '
#                          f'output node: {input_graph.ops[node_name].outputs}| '
#                          f'output tensor: {input_graph.ops[node_name].parameters["values"]["tensor"].size} }}')
#                 dot.node(node_name, label=label, shape='record', width='1.6', height='0.8',
#                      fontsize="12",
#                      style='rounded,filled',
#                      fillcolor='lightblue', color='blue', fontname='Consolas')

#     # 为输入输出节点创建点状节点
#     for node_name in temp_nodes:
#         dot.node(node_name, shape='point', width='0.05', height='0.05', fontsize='8', fontname='Consolas')

#     # 创建节点间的连接边
#     for item in nodes_data:
#         node_name = item
#         for i in input_graph.ops[item].inputs:
#             dot.edge(str(i), node_name)
#         for i in input_graph.ops[item].outputs:
#             dot.edge(node_name, str(i))

#     # 设置图形方向为从上到下
#     dot.attr(rankdir='TB')
    
#     # 保存图形文件
#     file_path = "./result/" + model_name + "/" + model_name + "_ops_graph"
#     dot.render(file_path, format='png', cleanup=True)

# def GraphGenerate(input_graph, model_name):
#     """
#     生成计算图的可视化图形
#     """
#     # 创建有向图对象
#     dot = Digraph()
    
#     # 提取所有操作节点名称
#     nodes_data = list(input_graph.ops.keys())
    
#     # 收集所有输入和输出节点名称用于绘制小圆点
#     temp_nodes = []
#     for item in nodes_data:
#         op = input_graph.ops[item]
#         for i in op.inputs:
#             temp_nodes.append(str(i))
#         for i in op.outputs:
#             temp_nodes.append(str(i))
    
#     # 去重处理
#     temp_nodes = list(set(temp_nodes))

#     # 为每个操作节点创建图形节点
#     for node_name in nodes_data:
#         op = input_graph.ops[node_name] # 获取算子对象
#         op_type = op.__class__.__name__ # 获取算子类型名称 (如 Conv, Relu, Constant)

#         has_tensor_info = False
#         tensor_data = None
        
#         if hasattr(op, "parameters") and isinstance(op.parameters, dict):
#             if "values" in op.parameters and "tensor" in op.parameters["values"]:
#                 tensor_data = op.parameters["values"]["tensor"]
#                 has_tensor_info = True
        
#         if not has_tensor_info:
#             label = (f'{{ {node_name} | '
#                      f'Type: {op_type} | ' 
#                      f'Output: {op.outputs} }}')
            
#             dot.node(node_name, label=label, shape='record', width='1.2', height='0.6',
#                      fontsize="10", style='filled,rounded', 
#                      fillcolor='lightgrey', color='gray', fontname='Consolas')
            
#         else:
#             if isinstance(tensor_data, list):
#                 sizes = [str(t.size) for t in tensor_data]
#                 size_str = ", ".join(sizes)
#             else:
#                 size_str = str(tensor_data.size)

#             label = (f'{{ Input: {op.inputs} | '
#                      f'{node_name} ({op_type}) | ' 
#                      f'Output: {op.outputs}| '
#                      f'Shape: [{size_str}] }}')
            
#             dot.node(node_name, label=label, shape='record', width='1.6', height='0.8',
#                      fontsize="12", style='filled,rounded', 
#                      fillcolor='lightblue', color='blue', fontname='Consolas')

#     for node_name in temp_nodes:
#         dot.node(node_name, shape='point', width='0.05', height='0.05', fontsize='8', fontname='Consolas')

#     for item in nodes_data:
#         node_name = item
#         op = input_graph.ops[item]
#         for i in op.inputs:
#             dot.edge(str(i), node_name)
#         for i in op.outputs:
#             dot.edge(node_name, str(i))

#     dot.attr(rankdir='TB')
    
#     import os
#     save_dir = "./result/" + model_name
#     if not os.path.exists(save_dir):
#         try:
#             os.makedirs(save_dir)
#         except OSError:
#             pass # 忽略目录已存在的并发错误
            
#     file_path = save_dir + "/" + model_name + "_ops_graph"
#     try:
#         dot.render(file_path, format='png', cleanup=True)
#         print(f"✅ 图形已保存: {file_path}.png")
#     except Exception as e:
#         print(f"❌ Graphviz 生成失败 (请确保已安装 Graphviz): {e}")

def GraphGenerate(input_graph, model_name):
    """
    生成计算图的可视化图形 (高性能优化版: SFDP 引擎)
    """
    print(f"   [GraphViz] 正在构建图结构 (模式: Op-to-Op, 引擎: sfdp)...")
    
    # === 关键修改 1: 切换引擎为 sfdp (适合大图) ===
    # overlap='false': 防止节点重叠
    # splines='true': 使用曲线连接 (如果太慢可改为 'false' 用直线)
    dot = Digraph(comment=model_name, engine='sfdp') 
    dot.attr(overlap='false', splines='true', sep='+10') 
    
    # 字体设置
    dot.attr('node', shape='record', style='filled,rounded', fontname='Consolas', fontsize='12')

    # ... (中间的 节点生成 和 连线逻辑 保持完全不变) ...
    # 为了方便，我把中间核心逻辑简写，请保留你之前代码中的 Step 1, 2, 3 部分
    
    # 1. 建立生产者映射
    tensor_producer_map = {}
    nodes_data = list(input_graph.ops.keys())
    for op_name in nodes_data:
        op = input_graph.ops[op_name]
        for out_tensor in op.outputs:
            tensor_producer_map[str(out_tensor)] = op_name

    # 2. 绘制算子节点
    print(f"   [GraphViz] 正在生成 {len(nodes_data)} 个算子节点...")
    for node_name in nodes_data:
        op = input_graph.ops[node_name]
        op_type = op.__class__.__name__

        # 尝试获取输出 Shape 信息
        shape_info = ""
        try:
            if hasattr(op, "parameters") and isinstance(op.parameters, dict):
                if "values" in op.parameters and "tensor" in op.parameters["values"]:
                    t_data = op.parameters["values"]["tensor"]
                    if isinstance(t_data, list):
                        shape_info = "\\n".join([f"Out{i}: {t.size}" for i, t in enumerate(t_data)])
                    elif hasattr(t_data, "size"):
                        shape_info = f"Out: {t_data.size}"
        except:
            pass
        
        if shape_info:
            label = f"{{ {node_name} | {op_type} | {shape_info} }}"
            fillcolor = 'lightblue'
        else:
            label = f"{{ {node_name} | {op_type} }}"
            fillcolor = 'lightgrey'

        dot.node(node_name, label=label, fillcolor=fillcolor)

    # 3. 绘制连线
    print(f"   [GraphViz] 正在计算连接关系...")
    drawn_edges = set()
    for op_name in nodes_data:
        op = input_graph.ops[op_name]
        for in_tensor in op.inputs:
            tensor_name = str(in_tensor)
            if tensor_name in tensor_producer_map:
                producer_name = tensor_producer_map[tensor_name]
                edge_key = (producer_name, op_name, tensor_name)
                if edge_key not in drawn_edges:
                    # sfdp 模式下，建议把连线 label 字体设小，或者干脆去掉 label
                    dot.edge(producer_name, op_name, fontsize="8", color="gray50") 
                    drawn_edges.add(edge_key)
            else:
                input_node_name = f"Input_{tensor_name}"
                if input_node_name not in drawn_edges:
                    dot.node(input_node_name, label=f"Input\\n{tensor_name}", shape='ellipse', fillcolor='lightgreen', style='filled')
                    drawn_edges.add(input_node_name)
                dot.edge(input_node_name, op_name, color="green")

    # 4. 保存与渲染
    save_dir = os.path.join("./result", model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
            
    file_path = os.path.join(save_dir, model_name + "_ops_graph")
    
    print(f"   [GraphViz] 正在渲染 (sfdp引擎)... 预计 10秒 内完成。")
    try:
        dot.render(file_path, format='pdf', cleanup=True) 
        print(f"✅ 可视化文件生成成功！请查看: {file_path}.pdf")
    except Exception as e:
        print(f"❌ 渲染失败: {e}")