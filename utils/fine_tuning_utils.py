import numpy as np
import random


def query_generation(support_set_label, test_loader, n_samples_per_class=1):
    support_set_label = [int(i) for i in support_set_label]
    test_x_total = test_loader.dataset.x_data_total
    test_y_total = test_loader.dataset.y_data_total
    query_x_total = np.empty(shape=(0, test_x_total.shape[1], test_x_total.shape[2]), dtype=np.float32)
    query_y_total = np.empty(shape=(0,), dtype=np.int32)

    for label in support_set_label:
        # query_data_indices = random.sample(test_x_total[test_y_total == label, :, :], n_samples_per_clas)
        query_x = np.array(random.sample(list(test_x_total[test_y_total == label, :, :]), n_samples_per_class))
        query_x_total = np.vstack((query_x_total, query_x))
        query_y = np.array([label for i in range(n_samples_per_class)], dtype=np.int32)
        query_y_total = np.hstack((query_y_total, query_y))

    return query_x_total, query_y_total


def model_parameter_printing(model):
    # 输出模型所有可训练参数
    print('\nThe fine-tuned parameters include:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    #输出模型所有参数值
    # for i in fine_tuned_model.named_parameters():
    #     print(i)
