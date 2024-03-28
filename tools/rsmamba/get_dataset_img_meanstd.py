import mmcv
import mmengine
import tqdm


def get_mean_std(item):
    img_folder, img_file = item
    img = mmcv.imread(f'{img_folder}/{img_file}', channel_order='rgb')
    mean = img.mean(axis=(0, 1))
    std = img.std(axis=(0, 1))
    return mean, std

def main():
    img_folder = '/path_to_data/NWPU-RESISC45'
    info_save_folder = 'datainfo/NWPU'
    mmengine.mkdir_or_exist(info_save_folder)
    img_files = mmengine.list_from_file(f'{info_save_folder}/train.txt')

    items = [(img_folder, img_file) for img_file in img_files]
    mean_std = mmengine.track_parallel_progress(get_mean_std, items, nproc=32)
    mean = [item[0] for item in mean_std]
    std = [item[1] for item in mean_std]
    mean = sum(mean) / len(mean)
    std = sum(std) / len(std)
    print(f'mean: {mean}, std: {std}')
    with open(f'{info_save_folder}/meanstd.txt', 'w') as f:
        f.write(f'mean: {mean}, std: {std}')


if __name__ == '__main__':
    main()
