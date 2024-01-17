
import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from assess import hist_sum, compute_metrics
# from pytorchtools import EarlyStopping

from index2one_hot import get_one_hot
from poly import adjust_learning_rate_poly

from models.ISDANet import MY_NET


from LEVIRdataset import LEVIRDataset
# from SYSUCDdataset import SYSUCDDataset
# from GZCDDdataset import GZCDDDataset



import warnings
warnings.filterwarnings('ignore')


# train_data = SYSUCDDataset(mode='train')
# train_data = GZCDDDataset(mode='train')
train_data = LEVIRDataset(mode='train')
data_loader = DataLoader(train_data, batch_size=8, shuffle=True)



# test_data = SYSUCDDataset(mode='test')
# test_data = GZCDDDataset(mode='test')
test_data = LEVIRDataset(mode='test')
test_data_loader = DataLoader(test_data, batch_size=8, shuffle=False)

Epoch = 200
lr = 0.0001
n_class = 2
F1_max = 0.85

root = r'C:\PycharmProjects\pytorch-from book\codes-paper2\results\xiaorong_v2_LE\heatmap_xiaorongcat'

net = MY_NET(2).cuda()

criterion = nn.BCEWithLogitsLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=5)


with open(root +'/train.txt', 'a') as f:
    for epoch in range(Epoch):
        # print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])

        torch.cuda.empty_cache()

        new_lr = adjust_learning_rate_poly(optimizer, epoch, Epoch, lr, 0.9)
        print('lr:', new_lr)

        _train_loss = 0

        _hist = np.zeros((n_class, n_class))

        net.train()
        for before, after, change in tqdm(data_loader, desc='epoch{}'.format(epoch), ncols=100):
            before = before.cuda()
            after = after.cuda()

            # ed_change = change.cuda()
            # ed_change = edge(ed_change)
            # lbl = torch.where(ed_change > 0.1, 1, 0)
            # plt.figure()
            # plt.imshow(lbl.data.cpu().numpy()[0][0], cmap='gray')
            # plt.show()
            # lbl = lbl.squeeze(dim=1).long().cpu()
            # lbl_one_hot = get_one_hot(lbl, 2).permute(0, 3, 1, 2).contiguous().cuda()

            change = change.squeeze(dim=1).long()
            change_one_hot = get_one_hot(change, 2).permute(0, 3, 1, 2).contiguous().cuda()

            optimizer.zero_grad()


            pred, aux1, aux2, aux3 = net(before, after)
            loss_pred = criterion(pred, change_one_hot)
            loss_aux_1 = criterion(aux1, change_one_hot)
            loss_aux_2 = criterion(aux2, change_one_hot)
            loss_aux_3 = criterion(aux3, change_one_hot)
            loss = loss_pred + loss_aux_1 + loss_aux_2 + loss_aux_3


            loss.backward()
            optimizer.step()
            _train_loss += loss.item()

            label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy()
            label_true = change.data.cpu().numpy()

            hist = hist_sum(label_true, label_pred, 2)

            _hist += hist

        # scheduler.step()

        miou, oa, kappa, precision, recall, iou, F1 = compute_metrics(_hist)

        trainloss = _train_loss / len(data_loader)

        print('Epoch:', epoch, ' |train loss:', trainloss, ' |train oa:', oa,  ' |train iou:', iou, ' |train F1:', F1)
        f.write('Epoch:%d|train loss:%0.04f|train miou:%0.04f|train oa:%0.04f|train kappa:%0.04f|train precision:%0.04f|train recall:%0.04f|train iou:%0.04f|train F1:%0.04f' % (
                epoch, trainloss, miou, oa, kappa, precision, recall, iou, F1))
        f.write('\n')
        f.flush()

        with torch.no_grad():
            with open(root + '/test.txt', 'a') as f1:

                torch.cuda.empty_cache()

                _test_loss = 0

                _hist = np.zeros((n_class, n_class))

                k = 0

                net.eval()
                for before, after, change in tqdm(test_data_loader, desc='epoch{}'.format(epoch), ncols=100):
                    before = before.cuda()
                    after = after.cuda()
                    change = change.squeeze(dim=1).long()
                    change_one_hot = get_one_hot(change, 2).permute(0, 3, 1, 2).contiguous().cuda()

                    pred, aux1, aux2, aux3 = net(before, after)

                    loss = criterion(pred, change_one_hot)

                    # loss_aux_1 = criterion(aux1, change_one_hot)
                    # loss_aux_2 = criterion(aux2, change_one_hot)
                    # loss_aux_3 = criterion(aux3, change_one_hot)
                    # loss = loss_pred + 0.4 * loss_aux_1 + 0.4 * loss_aux_2 + 0.4 * loss_aux_3

                    _test_loss += loss.item()

                    label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy()
                    label_true = change.data.cpu().numpy()

                    hist = hist_sum(label_true, label_pred, 2)

                    _hist += hist

                miou, oa, kappa, precision, recall, iou, F1 = compute_metrics(_hist)

                testloss = _test_loss / len(test_data_loader)

                print('Epoch:', epoch, ' |test loss:', testloss, ' |test oa:', oa,  ' |test iou:', iou, ' |test F1:', F1)
                f1.write('Epoch:%d|test loss:%0.04f|test miou:%0.04f|test oa:%0.04f|test kappa:%0.04f|test precision:%0.04f|test recall:%0.04f|test iou:%0.04f|test F1:%0.04f' % (
                    epoch, testloss, miou, oa, kappa, precision, recall, iou, F1))
                f1.write('\n')
                f1.flush()
        # scheduler.step(testloss)


            if F1 > F1_max:
                # save_path = args.summary_path+args.dir_name+'/checkpoints/'+'miou_{:.6f}.pth'.format(miou)
                # torch.save(model.state_dict(), save_path)

                save_path = root + '/F1_{:.4f}_iou_{:.4f}_epoch_{}.pth'.format(F1, iou, epoch)
                torch.save(net.state_dict(), save_path)

                # torch.save(net, root + 'F1_{:.4f}_epoch_{}.pth'.format(F1, epoch))
                F1_max = F1
