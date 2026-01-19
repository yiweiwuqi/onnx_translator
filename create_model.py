# # # create_model.py
# # import torch
# # import torch.nn as nn
# # import os

# # class ComprehensiveModel(nn.Module):
# #     """
# #     一个包含 Relu, Abs, Add, Cos 的测试模型。
# #     它接收两个输入。
# #     """
# #     def __init__(self):
# #         super(ComprehensiveModel, self).__init__()
# #         self.relu = nn.ReLU()

# #     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
# #         """
# #         定义模型的前向传播路径:
# #         z = Cos( Relu(x) + Abs(y) )
# #         """
# #         # 1. 应用 ReLU
# #         x_relu = self.relu(x)
        
# #         # 2. 应用 Abs
# #         y_abs = torch.abs(y)
        
# #         # 3. 应用 Add (会产生广播)
# #         # 假设 x 的形状是 (1, 3, 32, 32)
# #         # 假设 y 的形状是 (1, 1, 1, 32)
# #         added = x_relu + y_abs
        
# #         # 4. 应用 Cos
# #         output = torch.cos(added)
        
# #         return output

# # # --- 主执行流程 ---
# # if __name__ == "__main__":
# #     # 1. 实例化模型并设置为评估模式
# #     model = ComprehensiveModel()
# #     model.eval()

# #     # 2. 创建两个符合广播规则的虚拟输入张量
# #     dummy_input_x = torch.randn(1, 3, 32, 32)
# #     dummy_input_y = torch.randn(1, 1, 1, 32) # 这个张量将会被广播

# #     # 3. 定义 ONNX 文件的名称
# #     onnx_dir = "onnx_model"
# #     if not os.path.exists(onnx_dir):
# #         os.makedirs(onnx_dir)
# #     onnx_file_name = os.path.join(onnx_dir, "model.onnx")

# #     # 4. 执行导出操作
# #     print(f"正在导出模型到 {onnx_file_name}...")
# #     torch.onnx.export(
# #         model,
# #         (dummy_input_x, dummy_input_y), # 传入元组作为多个输入
# #         onnx_file_name,
# #         input_names=["input_x", "input_y"],  # 两个输入节点名称
# #         output_names=["output"],             # 输出节点名称
# #         opset_version=17,
# #         verbose=True
# #     )

# #     print(f"\n✅ 模型已成功导出为 '{onnx_file_name}'！")

# import torch
# import torch.nn as nn
# import torch.onnx
# import torch.nn.functional as F
# import os

# class AllOperatorModel(nn.Module):
#     def __init__(self):
#         super(AllOperatorModel, self).__init__()
#         # --- 1. 定义层 ---
#         self.conv2d = nn.Conv2d(4, 4, 3, padding=1)
        
#         # [Fix] 输入维度修正为 4096 (4*32*32)，适配输入 shape
#         self.linear = nn.Linear(4096, 32)
        
#         self.bn = nn.BatchNorm2d(4)
#         self.ln = nn.LayerNorm([4, 32, 32])
#         self.in_norm = nn.InstanceNorm2d(4)
#         self.gn = nn.GroupNorm(2, 4)
#         self.embed = nn.Embedding(10, 4)
#         self.pixel_shuffle = nn.PixelShuffle(2) 
#         self.pixel_unshuffle = nn.PixelUnshuffle(2)
        
#         # 激活函数列表
#         self.activations = nn.ModuleList([
#             nn.ReLU(), nn.LeakyReLU(0.1), nn.Sigmoid(), nn.Tanh(), 
#             nn.ELU(), nn.SELU(), nn.CELU(), nn.Softplus(), nn.Softsign(), 
#             nn.Hardswish(), nn.GELU(), nn.Mish(), nn.Hardsigmoid(), 
#             nn.Threshold(0.0, 0.0), # ThresholdedRelu
#             nn.LogSoftmax(dim=1), nn.Softmax(dim=1)
#         ])
        
#         # 池化层
#         self.maxpool = nn.MaxPool2d(2)
#         self.avgpool = nn.AvgPool2d(2)
#         self.lppool = nn.LPPool2d(2, 2)
#         self.global_avg = nn.AdaptiveAvgPool2d(1)
#         self.global_max = nn.AdaptiveMaxPool2d(1)

#         # [Fix] 预先注册 idx buffer，解决 TracerWarning
#         self.register_buffer('idx_buffer', torch.zeros((1, 4, 32, 32), dtype=torch.int64))

#     def forward(self, x):
#         # x shape: [1, 4, 32, 32]
#         results = []
        
#         # --- 1. 基础算子 ---
#         x_conv = self.conv2d(x) 
#         x_bn = self.bn(x_conv) 
#         x_in = self.in_norm(x_conv)
#         x_ln = self.ln(x_conv)
#         x_gn = self.gn(x_conv)
        
#         x_flat = torch.flatten(x_conv, 1) 
#         x_gemm = self.linear(x_flat) 
#         results.append(x_gemm.sum().view(1))

#         # --- 2. 激活函数 ---
#         for act in self.activations:
#             results.append(act(x).sum().view(1))

#         # --- 3. 池化 ---
#         results.append(self.maxpool(x).sum().view(1))
#         results.append(self.avgpool(x).sum().view(1))
#         results.append(self.lppool(x).sum().view(1))
#         results.append(self.global_avg(x).sum().view(1))
#         results.append(self.global_max(x).sum().view(1))

#         # --- 4. 数学运算 (Math - Unary) ---
#         results.append(torch.abs(x).sum().view(1))
#         results.append(torch.neg(x).sum().view(1))
#         results.append(torch.reciprocal(x + 1.0).sum().view(1))
#         results.append(torch.ceil(x).sum().view(1))
#         results.append(torch.floor(x).sum().view(1))
#         results.append(torch.round(x).sum().view(1))
#         results.append(torch.sign(x).sum().view(1))
#         results.append(torch.sqrt(torch.abs(x)).sum().view(1))
#         results.append(torch.exp(x * 0.1).sum().view(1))
#         results.append(torch.log(torch.abs(x) + 1.0).sum().view(1))
#         results.append(torch.sin(x).sum().view(1))
#         results.append(torch.cos(x).sum().view(1))
#         results.append(torch.tan(x).sum().view(1))
#         results.append(torch.asin(torch.clamp(x, -0.9, 0.9)).sum().view(1))
#         results.append(torch.acos(torch.clamp(x, -0.9, 0.9)).sum().view(1))
#         results.append(torch.atan(x).sum().view(1))
#         results.append(torch.erf(x).sum().view(1))
        
#         # [Fix] 屏蔽导致 Opset 17 导出报错的双曲函数
#         # results.append(torch.sinh(x).sum().view(1))
#         # results.append(torch.cosh(x).sum().view(1))
#         # results.append(torch.asinh(x).sum().view(1))
#         # results.append(torch.acosh(torch.abs(x) + 1.0).sum().view(1))
#         # results.append(torch.atanh(torch.clamp(x, -0.9, 0.9)).sum().view(1))

#         # --- 5. 数学运算 (Math - Binary) ---
#         y = x * 0.5
#         results.append(torch.add(x, y).sum().view(1))
#         results.append(torch.sub(x, y).sum().view(1))
#         results.append(torch.mul(x, y).sum().view(1))
#         results.append(torch.div(x, y + 1.0).sum().view(1))
#         results.append(torch.pow(torch.abs(x), 2.0).sum().view(1))
#         results.append(torch.fmod(x, 2.0).sum().view(1))
#         results.append(torch.max(x, y).sum().view(1))
#         results.append(torch.min(x, y).sum().view(1))
        
#         x_int = x.to(torch.int32)
#         # [Fix] 屏蔽导致 Opset 17 导出报错的位运算
#         # results.append(torch.bitwise_and(x_int, x_int).float().sum().view(1))
#         # results.append(torch.bitwise_or(x_int, x_int).float().sum().view(1))
#         # results.append(torch.bitwise_xor(x_int, x_int).float().sum().view(1))
#         # results.append(torch.bitwise_not(x_int).float().sum().view(1))

#         # BitShift (Left/Right) - Opset 11+ 支持
#         results.append((x_int << 1).float().sum().view(1))
#         results.append((x_int >> 1).float().sum().view(1))

#         # --- 6. 逻辑运算 ---
#         b1 = torch.eq(x, y)
#         b2 = torch.gt(x, y)
#         b3 = torch.lt(x, y)
#         results.append(b1.float().sum().view(1))
#         results.append(b2.float().sum().view(1))
#         results.append(b3.float().sum().view(1))
#         results.append(torch.isnan(x).float().sum().view(1))
#         results.append(torch.isinf(x).float().sum().view(1))
        
#         l1 = torch.logical_and(b1, b2)
#         l2 = torch.logical_or(b1, b2)
#         l3 = torch.logical_xor(b1, b2)
#         l4 = torch.logical_not(b1)
#         results.append(l1.float().sum().view(1))

#         results.append(torch.where(b1, x, y).sum().view(1))

#         # --- 7. 归约 ---
#         results.append(torch.mean(x).view(1))
#         results.append(torch.sum(x).view(1))
#         results.append(torch.prod(torch.abs(x)+0.1).view(1))
#         results.append(torch.norm(x, p=1, dim=1).sum().view(1)) 
#         results.append(torch.norm(x, p=2, dim=1).sum().view(1))
#         results.append(torch.logsumexp(x, dim=1).sum().view(1))
        
#         results.append(torch.argmax(x, dim=1).float().sum().view(1))
#         results.append(torch.argmin(x, dim=1).float().sum().view(1))

#         # --- 8. 形状操作 ---
#         x_reshaped = x.view(1, 4, 1024)
#         x_trans = torch.transpose(x_reshaped, 1, 2)
#         results.append(x_trans.sum().view(1))
        
#         x_slice = x[:, 0:1, :, :]
#         results.append(x_slice.expand(1, 4, 32, 32).sum().view(1))
#         results.append(x_slice.repeat(1, 4, 1, 1).sum().view(1))

#         x_cat = torch.cat([x, x], dim=1)
#         x_split = torch.chunk(x_cat, 2, dim=1)
#         results.append(x_split[0].sum().view(1))
        
#         x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
#         results.append(x_pad.sum().view(1))
        
#         # [Fix] 使用 register_buffer 的 idx
#         idx = self.idx_buffer
#         gathered = torch.gather(x, 1, torch.clamp(idx, 0, 3)) 
#         results.append(gathered.sum().view(1))
        
#         scattered = torch.scatter(x, 1, torch.clamp(idx, 0, 3), torch.ones_like(x))
#         results.append(scattered.sum().view(1))
        
#         results.append(torch.nonzero(x).float().sum().view(1))
#         results.append(F.interpolate(x, scale_factor=2, mode='nearest').sum().view(1))
#         results.append(torch.einsum('bchw,bchw->bchw', x, x).sum().view(1))
        
#         val_topk, ind_topk = torch.topk(x.flatten(), 3)
#         results.append(val_topk.sum().view(1))
#         results.append(torch.cumsum(x.flatten(), 0).sum().view(1))
#         results.append(torch.tril(x[0,0]).sum().view(1))
#         results.append(torch.triu(x[0,0]).sum().view(1))
        
#         x_d2s = self.pixel_shuffle(x)
#         x_s2d = self.pixel_unshuffle(x_d2s)
#         results.append(x_s2d.sum().view(1))
        
#         indices = torch.arange(0, 4, device=x.device).long()
#         results.append(F.one_hot(indices, 4).float().sum().view(1))
        
#         # 汇总
#         final_output = torch.cat(results, dim=0).sum()
#         return final_output

# def export_full_graph():
#     model = AllOperatorModel()
#     model.eval()
#     dummy_input = torch.randn(1, 4, 32, 32)
#     output_dir = "./onnx_model"
#     if not os.path.exists(output_dir): os.makedirs(output_dir)
#     output_path = os.path.join(output_dir, "model.onnx")

#     print(f"🚀 正在导出全量算子模型 (Skipping Sinh/Cosh/Bitwise) 到: {output_path}")
    
#     torch.onnx.export(
#         model,
#         dummy_input,
#         output_path,
#         opset_version=17, 
#         input_names=['input'],
#         output_names=['output'],
#         do_constant_folding=False
#     )
#     print("✅ 导出成功！")

# if __name__ == "__main__":
#     export_full_graph()

import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F
import os

class FinalSupportedModel(nn.Module):
    """
    最终测试模型。
    ✅ 保留: Conv, BN, LN, IN, AvgPool, LeakyReLU, 数学运算, 形状变换
    ❌ 移除: Trilu (库不支持), GroupNorm (Opset 17 分解导致形状推断异常)
    """
    def __init__(self):
        super(FinalSupportedModel, self).__init__()
        
        # --- 1. 卷积与归一化 ---
        self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
        
        # 归一化 (保留支持良好的)
        self.bn = nn.BatchNorm2d(8)
        self.in_norm = nn.InstanceNorm2d(8)
        self.ln = nn.LayerNorm([8, 32, 32]) 
        # self.gn = nn.GroupNorm(4, 8) # [已移除] Opset 17 下会导致复杂的 Reshape 问题
        
        # --- 2. 激活函数 ---
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU()
        self.selu = nn.SELU()
        self.softplus = nn.Softplus()
        self.gelu = nn.GELU()
        
        # --- 3. 池化 ---
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- 4. 全连接 ---
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 128) # 8*8*8=512
        
        # Buffer
        self.register_buffer('dummy_idx', torch.tensor([0], dtype=torch.int64))

    def forward(self, x):
        # x: [1, 4, 32, 32]
        results = []
        
        # === 1. 多分支特征提取 ===
        # 分支 A: Conv -> BN -> ReLU
        x1 = self.conv1(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        
        # 分支 B: Conv -> IN -> LeakyReLU
        x2 = self.conv1(x)
        x2 = self.in_norm(x2)
        x2 = self.leaky_relu(x2)
        
        # 分支 C: Conv -> LN -> Tanh
        x3 = self.conv1(x)
        x3 = self.ln(x3)
        x3 = self.tanh(x3)
        
        # [已移除 GN 分支，直接复用卷积结果做 Sigmoid]
        x4 = self.conv1(x)
        x4 = self.sigmoid(x4)
        
        # === 2. 融合 ===
        # 此时所有分支 shape 均为 [1, 8, 32, 32]
        out = x1 + x2 + x3 + x4
        
        # === 3. 后处理 ===
        out_max = self.maxpool(out) # 16x16
        out_avg = self.avgpool(out_max) # 8x8
        
        flat = self.flatten(out_avg) # [1, 512]
        fc_out = self.fc(flat)       # [1, 128]
        results.append(fc_out.sum().view(1))
        
        # === 4. 形状变换验证 ===
        reshaped = fc_out.view(1, 8, 16) 
        transposed = reshaped.transpose(1, 2)
        results.append(transposed.sum().view(1))
        
        # === 5. 数学运算 ===
        y = x * 0.5
        results.append(torch.add(x, y).sum().view(1))
        results.append(torch.sub(x, y).sum().view(1))
        results.append(torch.mul(x, y).sum().view(1))
        results.append(torch.div(x, y + 1.0).sum().view(1))
        
        results.append(torch.abs(x).sum().view(1))
        results.append(torch.pow(torch.abs(x), 2).sum().view(1))
        results.append(torch.sin(x).sum().view(1))
        results.append(torch.clamp(x, -0.5, 0.5).sum().view(1))

        # === 6. 其他操作 ===
        cat_res = torch.cat([x, x], dim=1) 
        split_res = torch.chunk(cat_res, 2, dim=1) 
        results.append(split_res[0].sum().view(1))
        
        expand_res = x[:, 0:1, :, :].expand(1, 4, 32, 32)
        results.append(expand_res.sum().view(1))
        
        results.append(self.global_avg(x).sum().view(1))
        
        # 汇总
        final_out = torch.cat(results, dim=0).sum()
        return final_out

def export_model():
    model = FinalSupportedModel()
    model.eval()
    dummy_input = torch.randn(1, 4, 32, 32)
    output_dir = "./onnx_model"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "model.onnx")
    
    print(f"🚀 正在导出最终模型 (移除 GroupNorm) 到: {output_path}")
    torch.onnx.export(
        model, dummy_input, output_path,
        opset_version=17,
        input_names=['input'], output_names=['output'],
        do_constant_folding=False
    )
    print("✅ 导出成功！")

if __name__ == "__main__":
    export_model()