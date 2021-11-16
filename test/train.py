from get_train_test import *


def load_all_dataset():
    src_train_dataset, tgt_train_dataset = get_train_test_dataset_1(dataset, 0.8)

    src_val_dataset, tgt_val_dataset = KnockDataset_val(root_dir, source_domain, SUPPORT_SET_LABEL)

    pair_wise_dataset = KnockDataset_pair(root_dir, support_label_set=SUPPORT_SET_LABEL)

    support_dataset = KnockDataset_test(root_dir, target_domain, SUPPORT_SET_LABEL)
    query_dataset = KnockDataset_test(root_dir, source_domain, SUPPORT_SET_LABEL)


def get_dataloader():
    # --------- DataLoader for Offline-Stage --------------
    # Source Train
    src_train_batch_sampler = BalancedBatchSampler(src_train_dataset.train_label, n_classes=src_train_n_classes,
                                                   n_samples=NUM_SAMPLES_PER_CLASS)
    src_train_loader = torch.utils.data.DataLoader(src_train_dataset, batch_sampler=src_train_batch_sampler, **kwargs)
    # Target Train
    tgt_train_batch_sampler = BalancedBatchSampler(tgt_train_dataset.train_label, n_classes=tgt_train_n_classes,
                                                   n_samples=NUM_SAMPLES_PER_CLASS)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_dataset, batch_sampler=tgt_train_batch_sampler, **kwargs)
    # Source Validation
    src_val_loader = torch.utils.data.DataLoader(src_val_dataset, batch_size=len(src_val_dataset))
    # Target Validation
    tgt_val_loader = torch.utils.data.DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset))
    # Pair-wise Train
    pair_wise_train_loader = torch.utils.data.DataLoader(pair_wise_dataset, batch_size=PAIR_WISE_BATCH)