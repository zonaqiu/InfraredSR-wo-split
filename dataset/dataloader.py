from torch.utils.data import DataLoader
from importlib import import_module


def get_dataloader(args):
    ### import module
    # 动态导入对象
    m = import_module('dataset.' + args.dataset.lower())  # cufed

    if (args.dataset == 'CUFED'):
        # 返回一个对象属性值
        data_train = getattr(m, 'TrainSet')(args)
        # 没有对数据集进行随机裁剪等操作，而且 参考图像的size 写死了。
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(5):
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader