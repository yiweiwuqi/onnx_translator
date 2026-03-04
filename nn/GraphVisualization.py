from graphviz import Digraph
import os

def GraphGenerate(input_graph, model_name, fast_mode=False):
    """
    生成计算图的可视化图形 (支持可选的极速模式)
    
    Args:
        input_graph: 输入的计算图对象
        model_name: 模型名称
        fast_mode: (bool) 是否开启极速渲染模式。默认 False。
                   - False: 标准渲染，布局最优化，但大图可能较慢。
                   - True:  极速渲染，限制迭代次数，速度快 5-10 倍，适合快速预览。
    """
    mode_info = "极速渲染 (nslimit=2)" if fast_mode else "标准渲染"
    print(f"   [GraphViz] 正在构建图结构 (布局: Top-Down, 模式: {mode_info})...")
    
    dot = Digraph(comment=model_name)
    
    dot.attr(rankdir='TB')
    
    if fast_mode:
        dot.attr(nslimit='2.0')
        dot.attr(nslimit1='2.0')

        dot.attr(mclimit='2.0')

        dot.attr(overlap='false', splines='true', sep='+10') 
    else:

        dot.attr(overlap='false', splines='true')

    dot.attr('node', shape='record', style='filled,rounded', fontname='Consolas', fontsize='12')

    valid_nodes = []
    skipped_nodes = set()
    
    for node_name, op in input_graph.ops.items():
        op_type = op.__class__.__name__
        if op_type == 'Constant':
            skipped_nodes.add(node_name)
            continue
        valid_nodes.append(node_name)

    print(f"   (已过滤 {len(skipped_nodes)} 个常量节点，保留 {len(valid_nodes)} 个计算节点)")

    tensor_producer_map = {}
    for op_name in valid_nodes:
        op = input_graph.ops[op_name]
        for out_tensor in op.outputs:
            tensor_producer_map[str(out_tensor)] = op_name

    for node_name in valid_nodes:
        op = input_graph.ops[node_name]
        op_type = op.__class__.__name__

        shape_info = ""
        try:
            if hasattr(op, "parameters") and isinstance(op.parameters, dict):
                if "values" in op.parameters and "tensor" in op.parameters["values"]:
                    t_data = op.parameters["values"]["tensor"]
                    if isinstance(t_data, list):
                        sizes = [str(t.size) for t in t_data]
                        shape_info = "\\n".join([f"Out{i}: {s}" for i, s in enumerate(sizes)])
                    elif hasattr(t_data, "size"):
                        shape_info = f"Out: {t_data.size}"
        except:
            pass
        
        safe_name = node_name.replace('{', '\\{').replace('}', '\\}').replace('|', '\\|')
        safe_info = shape_info.replace('{', '\\{').replace('}', '\\}').replace('|', '\\|')

        if safe_info:
            label = f"{{ {safe_name} | {op_type} | {safe_info} }}"
            fillcolor = 'lightblue'
        else:
            label = f"{{ {safe_name} | {op_type} }}"
            fillcolor = 'lightgrey'

        dot.node(node_name, label=label, fillcolor=fillcolor)
        
    drawn_edges = set()
    for op_name in valid_nodes:
        op = input_graph.ops[op_name]
        for in_tensor in op.inputs:
            tensor_name = str(in_tensor)
            
            if tensor_name in tensor_producer_map:
                producer_name = tensor_producer_map[tensor_name]
                if producer_name in valid_nodes:
                    edge_key = (producer_name, op_name, tensor_name)
                    if edge_key not in drawn_edges:
                        dot.edge(producer_name, op_name)
                        drawn_edges.add(edge_key)
            else:
                # 处理外部输入
                input_node_name = f"Input_{tensor_name}"
                input_node_name_clean = input_node_name.replace(':', '_').replace('[', '_').replace(']', '_')
                
                if input_node_name_clean not in drawn_edges:
                    dot.node(input_node_name_clean, label=f"{tensor_name}", shape='ellipse', fillcolor='lightgreen', style='filled', fontsize='10')
                    drawn_edges.add(input_node_name_clean)
                
                dot.edge(input_node_name_clean, op_name, color="green")

    save_dir = os.path.join("./result", model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
            
    file_path = os.path.join(save_dir, model_name + "_ops_graph")
    
    print(f"   [GraphViz] 正在渲染 SVG (wait)...")
    try:
        path = dot.render(file_path, format='svg', cleanup=True)
        print(f"✅ 可视化文件生成成功！请查看: {path}")
    except Exception as e:
        print(f"❌ 渲染失败: {e}")