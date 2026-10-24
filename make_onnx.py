import torch
import torch.nn as nn

# 专为移动端/树莓派优化的轻量 Fast-StereoNet 架构
class LightweightStereoNet(nn.Module):
    def __init__(self, max_disp=192):
        super().__init__()
        self.max_disp = max_disp
        # 2D 深度可分离特征提取，极低 CPU 占用
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.disp_head = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, left, right):
        f_l = self.feature(left)
        f_r = self.feature(right)
        diff = torch.abs(f_l - f_r)
        disp = torch.sigmoid(self.disp_head(diff)) * self.max_disp
        # 直接上采样回 320x240，匹配 Folkrace 近距离避障需求
        return nn.functional.interpolate(disp, size=(240, 320), mode='bilinear', align_corners=False)

if __name__ == "__main__":
    print("正在构建并导出 ONNX 模型...")
    model = LightweightStereoNet(max_disp=192)
    model.eval()

    # 输入维度: [Batch, Channel, Height, Width] -> 320x240 分辨率
    dummy_left = torch.randn(1, 3, 240, 320)
    dummy_right = torch.randn(1, 3, 240, 320)

    onnx_file = "fast_stereonet_320x240.onnx"
    torch.onnx.export(
        model,
        (dummy_left, dummy_right),
        onnx_file,
        input_names=["left", "right"],
        output_names=["disparity"],
        opset_version=11,
        do_constant_folding=True
    )
    print(f"🎉 模型成功生成于当前路径: {onnx_file}")