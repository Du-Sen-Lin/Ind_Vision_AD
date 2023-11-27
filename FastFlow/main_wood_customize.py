import argparse
import os

import torch
import yaml
from ignite.contrib import metrics

import constants as const
import dataset
import fastflow
import utils


def build_train_data_loader(args, config):
    train_dataset = dataset.CustomDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    test_dataset = dataset.CustomDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_once(dataloader, model):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach() # detach阻断反向传播，返回值仍为tensor
        if len(targets.shape) != 4:
            outputs = outputs.view(outputs.shape[0], -1)
            outputs, _ = torch.max(outputs, dim=1) # 输出: max 维度求得的最大值 max_indices 维度求得的最大值的索引 (模块项目中code错误, 导致auroc异常)
        # outputs = outputs.flatten()
        # targets = targets.flatten()
        # print(f"@@ 评估 outputs: {outputs.shape} type: {outputs.dtype}")
        # print(f"@@ 评估 targets: {targets.shape} type: {targets.dtype}")
        outputs = outputs.flatten()
        targets = targets.flatten().type(torch.int32)
        # print(f"@@ 评估 outputs: {outputs.shape} , {outputs[:6]},  type: {outputs.dtype}")
        # print(f"@@ 评估 targets: {targets.shape} , {targets[:6]},  type: {targets.dtype}")
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))


def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    print(f"@@ 模型保存路径: {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    # 模型创建
    model = build_model(config)
    optimizer = build_optimizer(model)

    print(f"@@ 参数 args: {args} \n \t config: {config}")
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    print(f"@@ 数据加载 train_dataloader: {len(train_dataloader)} test_dataloader: {len(test_dataloader)}")
    model.cuda()

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model)
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    eval_once(test_dataloader, model)


def parse_args():
    """
    ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    name or flags: 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo
    action: 当参数在命令行中出现时使用的动作基本类型
    nargs: 命令行参数应当消耗的数目
    const: 被一些 action 和 nargs 选择所需求的常数。
    default: 当参数未在命令行中出现并且也不存在于命名空间对象时所产生的值
    type: 命令行参数应当被转换成的类型
    choices: 可用的参数的容器
    required: 此命令行选项是否可省略 （仅选项可用), true, 表示不可省略, 如果没有这个参数, 会抛出异常。
    help: 一个此选项作用的简单描述
    metavar: 在使用方法消息中使用的参数值示例。
    dest: 被添加到 parse_args() 所返回对象上的属性名。
    """
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    # wide_resnet50_2.yaml resnet18.yaml
    parser.add_argument(
        "-cfg", "--config", default="/root/project/wood/ort/trt/research/wdcv/demo/anomaly_detection/FastFlow/configs/resnet18.yaml", type=str, help="path to config file"
    )
    parser.add_argument("--data", default=r'/root/dataset/wood/ort/trt/Research/industry_data', type=str, help="path to data folder")
    parser.add_argument(
        "-cat",
        "--category",
        default="carpet_in",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        print(f"@@ eval @@")
        evaluate(args)
    else:
        print(f"@@ train @@")
        train(args)

# data
"""
    data:
    ├── test
    │   ├── good
    │   |── ng1
    |   └── ng2 ...
    └── train
        └── good
"""

"""
carpet_in
GM06_08
"""
# train
# !python main_wood_customize.py
# eval
# !python main.py --eval -ckpt _fastflow_experiment_checkpoints/exp[index]/[epoch#].pt
