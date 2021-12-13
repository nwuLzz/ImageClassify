from torchvision import transforms, datasets


# 作者给出的标准化方法
def _norm_advprop(img):
    return img * 2.0 - 1.0


def build_transform(dest_image_size):
    # 正则化，将图像的每个通道的数据映射到同一区间
    # 为了保证数据集中所有的图像分布都相似，在训练时更容易收敛，既加快了训练速度，又提升了训练效果
    normalize = transforms.Lambda(_norm_advprop)

    if not isinstance(dest_image_size, tuple):
        dest_image_size = (dest_image_size, dest_image_size)
    else:
        dest_image_size = dest_image_size

    # # 随机变换
    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop(dest_image_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    # 非随机变换，保证每次预测结果都是一致的
    transform = transforms.Compose([
        transforms.Resize(dest_image_size),
        transforms.CenterCrop(dest_image_size),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        normalize
    ])

    return transform


def build_data_set(dest_image_size, data):
    transform = build_transform(dest_image_size)
    dataset = datasets.ImageFolder(data, transform=transform, target_transform=None)

    return dataset


def main(data):
    dataset = build_data_set(224, data)


if __name__ == '__main__':
    data = './data/train'
    main(data)
