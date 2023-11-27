import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const
batch_id = 0


def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        print(f"@@ in_channels: {in_channels}, out_channels: {out_channels}")
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    """
    for resner18.yaml:
        input_chw:[64 64 64]; [128, 32, 32]; [256, 16, 16]
        conv3x3_only: 对于AllInOneBlock [0 - flow_steps]: True, 都是conv3x3; False, 对于奇数AllInOneBlock,conv1x1
        hidden_ratio: 1.0, flow模块第一个卷积的输出通道: in_channels * hidden_ratio
        flow_steps: flow层数
    """
    nodes = Ff.SequenceINN(*input_chw)
    print(f"nodes: {nodes}")
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        # append() 方法: 将一个可逆块从FrEIC.modules 附加到网络
        # “permute_soft=True”在处理 >512 维度时非常慢。
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Module):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
    ):
        super(FastFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        # ViT
        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        # ResNet18 和 Wide-ResNet50-2
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm norms结果参数需要训练更新
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )
        # backbone 充当特征提取器，不需要训练，不需要更新参数 requires_grad==false
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # flows 模块
        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

    def forward(self, x):
        # ---------调试--------------
        temp_x = x
        # ----------------------------

        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            print(f"@@ vision_transformer.")
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            print(f"@@ Cait.")
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            print(f"@@ other forward (resnet18/wide_resnet50_2).")
            features = self.feature_extractor(x) # x shape: torch.Size([32, 3, 256, 256]) 
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        loss = 0
        outputs = []
        for i, feature in enumerate(features):
            # 特征传递给 SequenceINN ; 获得转换后的变量output 和 对数雅可比行列式(log Jacobian determinant)
            # 雅可比行列式: 
            output, log_jac_dets = self.nf_flows[i](feature)
            # 计算模型的负对数似然: 使用标准正态先验分布
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            # print(f"@@ output_sum: {0.5 * torch.sum(output**2, dim=(1, 2, 3))}, \n log_jac_dets: {log_jac_dets}")
            outputs.append(output)
        ret = {"loss": loss}

        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                a_map = F.interpolate(
                    -prob,
                    size=[self.input_size, self.input_size],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            print(f"@@ forward eval add ret anomaly_map, anomaly_map shape: {anomaly_map.shape}")

            # ---------调试--------------
            global batch_id
            batch_id += 1
            import cv2
            for i in range(temp_x.shape[0]): # to tensor + Normalize后的图
                save_out_temp = temp_x[i] # c h w
                save_out_temp = save_out_temp.permute(1, 2, 0)
                print(f"@@ save_out_temp: {save_out_temp.shape}")
                save_out_temp = save_out_temp.cpu().detach().numpy().astype('uint8')
                save_path = './temp_forward_images/'  + str(batch_id) + '_'+ str(i) + '_ori.jpg'
                cv2.imwrite(save_path, save_out_temp) 
            # ----------------------------             
            # ---------调试--------------
            save_out = torch.exp(anomaly_map) * 255
            for i in range(save_out.shape[0]): # 异常热力图
                save_out_temp = save_out[i][0]
                print(f"@@ save_out_temp: {save_out_temp.shape}")
                save_out_temp = save_out_temp.cpu().detach().numpy().astype('uint8')
                save_path = './temp_forward_images/' + str(batch_id) + '_' + str(i) + '.jpg'
                cv2.imwrite(save_path, save_out_temp)
            # -----------------------------

            ret["anomaly_map"] = anomaly_map
        return ret
