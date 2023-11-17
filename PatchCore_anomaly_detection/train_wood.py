import os
import glob
import shutil

import numpy as np
from PIL import Image
import cv2

from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score
from sampling_methods.kcenter_greedy import kCenterGreedy
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import faiss


def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)
    return dist


class NN():
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        dist = torch.cdist(x, self.train_pts, self.p)
        knn = dist.topk(self.k, largest=False)
        return knn

def prep_dirs(root):
    # make embeddings dir
    embeddings_path = os.path.join('./', 'embeddings', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    return embeddings_path, sample_path, source_code_save_path

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
            # print(f"#### dataset train: {self.img_path}")
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
            # print(f"#### dataset test: {self.img_path}, \t {self.gt_path}")
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1
        # print(f"#### load dataset: {self.img_paths} \n # {self.gt_paths} \n # {self.labels} \n # {self.types}")

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        print(f"### defect_types: {defect_types}")
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    
    

class PatchCore(pl.LightningModule):
    def __init__(self, hparams):
        super(PatchCore, self).__init__()

        # pl保存超参数
        self.save_hyperparameters(hparams)

        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)

        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        # https://github.com/pytorch/vision/issues/4156
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        # 预训练模型 Backbone 提取图像特征
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        # 不需要训练，不需要更新参数 requires_grad==false
        for param in self.model.parameters():
            param.requires_grad = False
        # 放弃了局部正常特征数据较少、偏向于分类任务的深层特征，采用第 [2, 3] 层特征作为图像特征
        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)
        # 损失函数：均方损失函数
        self.criterion = torch.nn.MSELoss(reduction='sum')
        # 初始化结果列表
        self.init_results_list()
        # 数据增强
        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])
        # ？？？
        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type):
        """
        保存热力图
        """
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        # print(f"self.sample_path: {self.sample_path}")
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        """
        训练数据集
        """
        print(f"#### train_dataloader")
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0)
        return train_loader

    def test_dataloader(self):
        print(f"#### test_dataloader")
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0)
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        print(f"#### on_train_start")
        self.model.eval() # to stop running_var move (maybe not critical)        
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        print(f"#### self.logger.log_dir: {self.logger.log_dir}, \n self.embedding_dir_path: {self.embedding_dir_path} \
              \n self.sample_path: {self.sample_path} \n self.source_code_save_path: {self.source_code_save_path}")
        self.embedding_list = []
    
    def on_test_start(self):
        print(f"#### on_test_start")
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss'))
        # cpu to gpu 在文件较大时（coreset_sampling_ratio为0.001 ok, 0.1时候报错）候会报错
        # 可能解决方法： https://www.pudn.com/news/6228d0ef9ddf223e1ad168ac.html
        # 尝试 conda install -c pytorch faiss-gpu cudatoolkit=11.0 
        # 测试（？指标居然都变得正常了和git一样了！淦！所以原来指标太差是faiss导致？）：
        # bottle 0.001 {'img_auc': 1.0, 'pixel_auc': 0.9779774579832146}
        # bottle 0.01  {'img_auc': 1.0, 'pixel_auc': 0.9808755876830869}
        # bottle 0.1   {'img_auc': 1.0, 'pixel_auc': 0.9815224723002607}
        # https://github.com/facebookresearch/faiss/issues/1405
        # torch版本：pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
        # 旧docker版本：sudo pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
            print(f"#### faiss index_cpu_to_gpu success.")
        self.init_results_list()
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        # print(f"#### training_step")
        x, _, _, _, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))

    def training_epoch_end(self, outputs): 
        print(f"#### training_epoch_end")
        # Memory Bank:将收集到的正常图像 Patch 特征放入 MemoryBank
        total_embeddings = np.array(self.embedding_list)
        # Random projection  随机投影  Johnson-Lindenstrauss 定理
        # 稀疏随机矩阵：使用稀疏随机矩阵，通过投影原始输入空间来降低维度。
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling 核心集二次抽样:稀疏采样 Reduce memory bank
        # 参考 https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py 
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset) 
        faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))


    def test_step(self, batch, batch_idx): # Nearest Neighbour Search 最近邻搜索
        # print(f"#### test_step")
        x, gt, label, file_name, x_type = batch
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))
        score_patches, _ = self.index.search(embedding_test , k=args.n_neighbors)
        anomaly_map = score_patches[:,0].reshape((28,28))
        N_b = score_patches[np.argmax(score_patches[:,0])]
        # 论文解释：
        '''
        To obtain s, we use a scaling w on s∗ to account for the behaviour of neighbouring patches: 
            If the memory bank features closest to the anomaly candidate mtest,∗, m∗, 
            is itself relatively far from neighbouring samples and thereby an already rare nominal occurence, 
            we increase the anomaly score
            如果内存库特征最接近异常候选 m^test,* , m^*, 本身距离近邻样本相对较远，因此是少见的正常发生，使用权重增加异常分数。 
            相当于计算了一个softmax
        '''
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
        score = w*max(score_patches[:,0]) # Image-level score
        gt_np = gt.cpu().numpy()[0,0].astype(int)
        # 将结果放大：匹配原始输入分辨率
        anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
        # 高斯平滑
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        # save images
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
        self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np*255, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        print(f"#### test_epoch_end")
        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        print(f"#### self.gt_list_img_lvl: {self.gt_list_img_lvl} \n self.pred_list_img_lvl: {self.pred_list_img_lvl}")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'/root/dataset/public/Research/DataSet/Anomaly_Detect/MVTec_AD/mvtec_anomaly_detection')
    parser.add_argument('--category', default='bottle')
    # num_epochs: patchCore 没有 PaDiM 那样的训练阶段（神经网络）,在代码中它只是提取特征而不更新参数。epochs=1
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--load_size', default=256)
    parser.add_argument('--input_size', default=224)
    # coreset_sampling_ratio 
    parser.add_argument('--coreset_sampling_ratio', default=0.1)
    parser.add_argument('--project_root_path', default=r'/root/project/wood/ort/trt/research/wdcv/demo/anomaly_detection/models/patchcore')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """
    https://www.pudn.com/news/62a7324e3934cd25af830293.html
    补丁嵌入向量(patch embedding vectors):所谓补丁，指的是像素；所谓嵌入，指的是将网络提取的不同特征组合到一块。都是封装的名词罢了。
    faiss: 全称(Facebook AI Similarity Search)是Facebook AI团队开源的针对聚类和相似性搜索库,为稠密向量提供高效相似度搜索和聚类，支持十亿级别向量的搜索，是目前较成熟的近似近邻搜索库。
    Faiss用C++编写,并提供与Numpy完美衔接的Python接口。除此以外,对一些核心算法提供了GPU实现。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=1)
    print(f"## 1、model ##")
    model = PatchCore(hparams=args)
    if args.phase == 'train':
        print(f"## 2、train ##")
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)

