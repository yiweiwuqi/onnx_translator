from graphviz import Digraph

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

def GraphGenerate(input_graph, model_name):
    """
    生成计算图的可视化图形
    """
    # 创建有向图对象
    dot = Digraph()
    
    # 提取所有操作节点名称
    nodes_data = list(input_graph.ops.keys())
    
    # 收集所有输入和输出节点名称用于绘制小圆点
    temp_nodes = []
    for item in nodes_data:
        op = input_graph.ops[item]
        for i in op.inputs:
            temp_nodes.append(str(i))
        for i in op.outputs:
            temp_nodes.append(str(i))
    
    # 去重处理
    temp_nodes = list(set(temp_nodes))

    # 为每个操作节点创建图形节点
    for node_name in nodes_data:
        op = input_graph.ops[node_name] # 获取算子对象
        op_type = op.__class__.__name__ # 获取算子类型名称 (如 Conv, Relu, Constant)

        has_tensor_info = False
        tensor_data = None
        
        if hasattr(op, "parameters") and isinstance(op.parameters, dict):
            if "values" in op.parameters and "tensor" in op.parameters["values"]:
                tensor_data = op.parameters["values"]["tensor"]
                has_tensor_info = True
        
        if not has_tensor_info:
            label = (f'{{ {node_name} | '
                     f'Type: {op_type} | ' 
                     f'Output: {op.outputs} }}')
            
            dot.node(node_name, label=label, shape='record', width='1.2', height='0.6',
                     fontsize="10", style='filled,rounded', 
                     fillcolor='lightgrey', color='gray', fontname='Consolas')
            
        else:
            if isinstance(tensor_data, list):
                sizes = [str(t.size) for t in tensor_data]
                size_str = ", ".join(sizes)
            else:
                size_str = str(tensor_data.size)

            label = (f'{{ Input: {op.inputs} | '
                     f'{node_name} ({op_type}) | ' 
                     f'Output: {op.outputs}| '
                     f'Shape: [{size_str}] }}')
            
            dot.node(node_name, label=label, shape='record', width='1.6', height='0.8',
                     fontsize="12", style='filled,rounded', 
                     fillcolor='lightblue', color='blue', fontname='Consolas')

    for node_name in temp_nodes:
        dot.node(node_name, shape='point', width='0.05', height='0.05', fontsize='8', fontname='Consolas')

    for item in nodes_data:
        node_name = item
        op = input_graph.ops[item]
        for i in op.inputs:
            dot.edge(str(i), node_name)
        for i in op.outputs:
            dot.edge(node_name, str(i))

    dot.attr(rankdir='TB')
    
    import os
    save_dir = "./result/" + model_name
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            pass # 忽略目录已存在的并发错误
            
    file_path = save_dir + "/" + model_name + "_ops_graph"
    try:
        dot.render(file_path, format='png', cleanup=True)
        print(f"✅ 图形已保存: {file_path}.png")
    except Exception as e:
        print(f"❌ Graphviz 生成失败 (请确保已安装 Graphviz): {e}")