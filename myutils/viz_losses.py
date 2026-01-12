import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    with open('Train_record_vggsound/ACL_ViT16_aclifa_1gpu/train_losses', 'rb') as f:
        train_loss_list = np.load(f, allow_pickle=True)

    with open('Train_record_vggsound/ACL_ViT16_aclifa_1gpu/validation_losses', 'rb') as f:
        validation_loss_list = np.load(f, allow_pickle=True)

    # train_loss_list = train_loss_list / 100
    # validation_loss_list = validation_loss_list / 30

    print(validation_loss_list)

    plt.plot(train_loss_list, label='Train')
    plt.plot(validation_loss_list, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(len(validation_loss_list)))
    plt.legend()
    plt.show()