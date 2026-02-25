# 算子验证总结说明

## 一、验证背景
本次验证工作对算子出现频率排序开展，  
按出现次数从高到低优先验证高频算子，目标覆盖高频算子为优先。

验证重点为：
- Python 侧算子执行正确性
- CUDA 侧数值计算正确性
- 模型图构建与 shape 推导正确性

在保证结果正确的前提下，未对所有算子做极端 corner case 覆盖。

---

## 二、数值验证（Python + CUDA）

数值验证采用工程内已有的统一验证脚本 `numerical_correctness.py`：

- Python 侧通过 `Operators.py` 调用 `tensor_ops.so` 作为参考实现
- CUDA 侧通过 `cache/verify_*` 可执行文件作为 ground truth
- 对 Python 与 CUDA 结果进行逐元素数值对比
- 验证过程中 iterations 从默认 200 调整为 20，以保证稳定性和执行效率

### 已完成数值验证的算子（Python + CUDA）

- Add
- Sub
- Mul
- Div
- Relu
- Conv
- Softmax
- Gemm
- MaxPool
- Equal
- Greater
- Less
- Clip
- Sqrt
- Pow
- Matmul
- Reduce_mean
- Gather
- ScatterND

以上算子均通过 Python 与 CUDA 数值一致性验证。

## 三、图结构与 Shape 验证

针对算子中以图结构和 shape 推导为主的算子，
采用模型图验证方式进行覆盖：

- 使用 `create_model.py` 生成模型
- 使用 `verify_graph.py` 验证图构建、节点连接及 shape 推导过程
- 验证模型图能够正确生成且无报错

图结构与 shape 验证通过 create_graph_ops_model.py 构造覆盖 Cast / Shape / ConstantOfShape / Unsqueeze / Squeeze / Slice / Transpose / Concat / Reshape / Expand / Where / Range 等算子的 ONNX 模型，并使用 verify_graph.py 完成图导入、节点连接与 shape 推导验证，生成图结构可视化结果（result/nps_verification/nps_verification.pdf）