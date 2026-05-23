import albumentations as A

def get_train_transforms(cfg, dataset_mean, dataset_std):
    return A.Compose([
        A.Resize(cfg.image_size, cfg.image_size),
        A.HorizontalFlip(p=cfg.p_horizontal),
        A.VerticalFlip(p=cfg.p_vertical),
        A.ShiftScaleRotate(
            shift_limit=cfg.shift_limit,
            scale_limit=cfg.scale_limit,
            rotate_limit=cfg.rotate_limit,
            p=cfg.p_shift_scale_rotate
        ),
        A.RandomBrightnessContrast(p=cfg.p_brightness),
        A.GaussNoise(p=cfg.p_gauss_noise),
        A.Normalize(
            mean=dataset_mean,
            std=dataset_std
        ),
        A.ToTensorV2(),
    ])

def get_val_transforms(cfg,dataset_mean, dataset_std):
    return A.Compose([
        A.Resize(cfg.image_size, cfg.image_size),
        A.Normalize(
            mean=dataset_mean,
            std=dataset_std
        ),
        A.ToTensorV2(),
    ])