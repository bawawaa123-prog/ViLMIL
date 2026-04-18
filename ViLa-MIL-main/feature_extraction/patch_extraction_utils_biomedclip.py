"""
BiomedCLIP特征提取工具函数
支持使用BiomedCLIP的图像编码器提取医学图像patch特征
"""

import os
import h5py
import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# BiomedCLIP依赖
from open_clip import create_model_from_pretrained, get_tokenizer

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_biomedclip_transforms():
    """
    BiomedCLIP图像预处理
    输入: 任意尺寸的RGB图像
    输出: 224x224 归一化张量
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # BiomedCLIP使用224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet统计量(BiomedCLIP通用)
            std=(0.229, 0.224, 0.225)
        )
    ])


class PatchesDataset(Dataset):
    """Patch数据集加载器"""
    def __init__(self, file_path, transform=None):
        file_names = os.listdir(file_path)
        self.imgs = [os.path.join(file_path, fn) for fn in file_names]
        self.coords = file_names
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        coord = self.coords[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, coord

    def __len__(self):
        return len(self.imgs)


def save_embeddings_biomedclip(model, fname, dataloader, overwrite=False):
    """
    使用BiomedCLIP提取并保存patch特征
    
    参数:
        model: BiomedCLIP模型
        fname: 输出文件名(不含.h5后缀)
        dataloader: Patch数据加载器
        overwrite: 是否覆盖已有文件
    """
    if os.path.isfile(f'{fname}.h5') and not overwrite:
        print(f"⏩ Skipping {fname}.h5 (already exists)")
        return None

    embeddings, coords, file_names = [], [], []

    # 提取图像特征
    for batch, coord in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            # BiomedCLIP图像编码器
            # 输出: [batch_size, 512] (ViT-B/16特征维度)
            image_features = model.encode_image(batch)
            embeddings.append(image_features.detach().cpu().numpy())
            file_names.append(coord)

    # 解析坐标(格式: slidename_x_y.png)
    for file_name in file_names:
        for coord in file_name:
            coord_parts = coord.rstrip('.png').split('_')
            if len(coord_parts) >= 2:
                try:
                    x = int(coord_parts[-2])  # 倒数第二个是x
                    y = int(coord_parts[-1])  # 最后一个是y
                    coords.append([x, y])
                except ValueError:
                    print(f"⚠️ Warning: 无法解析坐标 {coord}, 跳过")
                    continue

    # 保存到HDF5
    embeddings = np.vstack(embeddings)
    coords = np.vstack(coords)

    print(f"💾 Saving {fname}.h5 | Features: {embeddings.shape}, Coords: {coords.shape}")

    with h5py.File(f'{fname}.h5', 'w') as f:
        f.create_dataset('features', data=embeddings, compression='gzip')
        f.create_dataset('coords', data=coords, compression='gzip')


def create_embeddings_biomedclip(
    embeddings_dir,
    model_path,
    dataset,
    batch_size,
    patch_datasets='path/to/patches'
):
    """
    批量提取BiomedCLIP特征
    
    参数:
        embeddings_dir: 输出目录
        model_path: BiomedCLIP模型路径(HF hub或本地)
        dataset: 数据集名称
        batch_size: 批量大小
        patch_datasets: Patch根目录
    """
    print(f"\n🔬 Extracting BiomedCLIP Features for '{dataset}'")
    print(f"📦 Model: {model_path}")
    print(f"🖼️ Device: {device}")
    
    # 加载BiomedCLIP模型
    try:
        model, preprocess = create_model_from_pretrained(model_path)
        tokenizer = get_tokenizer(model_path)
        print("✅ BiomedCLIP model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load BiomedCLIP: {e}")
        print("💡 Trying to install open_clip_torch...")
        os.system("pip install open_clip_torch")
        model, preprocess = create_model_from_pretrained(model_path)
        tokenizer = get_tokenizer(model_path)
    
    model = model.to(device)
    model.eval()
    
    # 使用自定义transforms(而非preprocess,以保持与原代码一致)
    eval_transforms = get_biomedclip_transforms()
    
    # 获取特征维度
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = model.encode_image(dummy_input)
        feature_dim = dummy_output.shape[1]
    print(f"📐 Feature Dimension: {feature_dim}")
    
    # 遍历所有WSI
    wsi_list = os.listdir(patch_datasets)
    print(f"🔍 Found {len(wsi_list)} WSI folders")
    
    for wsi_name in tqdm(wsi_list, desc="Processing WSIs"):
        wsi_patch_dir = os.path.join(patch_datasets, wsi_name)
        
        # 跳过非目录文件
        if not os.path.isdir(wsi_patch_dir):
            continue
        
        # 创建数据加载器
        dataset_obj = PatchesDataset(wsi_patch_dir, transform=eval_transforms)
        
        # Windows下num_workers=0
        num_workers = 0 if os.name == 'nt' else 4
        dataloader = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # 保存特征
        fname = os.path.join(embeddings_dir, wsi_name)
        save_embeddings_biomedclip(model, fname, dataloader)
    
    print(f"\n✨ All features saved to: {embeddings_dir}")


def extract_text_features_biomedclip(
    text_prompts,
    model_path='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    output_path='text_features_biomedclip.pt'
):
    """
    使用BiomedCLIP提取文本特征(可选功能)
    
    参数:
        text_prompts: 文本提示列表 e.g., ["adenocarcinoma histopathology", ...]
        model_path: BiomedCLIP模型路径
        output_path: 输出文件路径
    
    返回:
        text_features: [num_prompts, 512] torch.Tensor
    """
    print(f"\n📝 Extracting Text Features with BiomedCLIP")
    
    model, _ = create_model_from_pretrained(model_path)
    tokenizer = get_tokenizer(model_path)
    
    model = model.to(device)
    model.eval()
    
    # 文本编码(使用tokenizer而非tokenize函数)
    texts = tokenizer(text_prompts).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(texts)
    
    # 归一化(可选,用于余弦相似度)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 保存
    torch.save(text_features, output_path)
    print(f"💾 Text features saved to: {output_path}")
    print(f"📐 Shape: {text_features.shape}")
    
    return text_features


# 示例用法
if __name__ == "__main__":
    # 测试图像特征提取
    test_prompts = [
        "adenocarcinoma histopathology",
        "normal tissue histopathology"
    ]
    
    text_features = extract_text_features_biomedclip(
        test_prompts,
        output_path='test_text_features.pt'
    )
    
    print(f"\n✅ Text feature extraction test passed!")
    print(f"Feature shape: {text_features.shape}")
