def make_loaders(n_train: int, n_test: int, seq_len: int, sigma: float, batch: int, num_workers: int = 2):
    device = torch.device("cpu")
    Xtr, ytr = generate_noisy_data(n_train, seq_len, sigma, device)
    Xte, yte = generate_noisy_data(n_test,  seq_len, sigma, device)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False  # set True if you ever move tensors to GPU
    )
    test_loader = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=batch,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False
    )
    return train_loader, test_loader
