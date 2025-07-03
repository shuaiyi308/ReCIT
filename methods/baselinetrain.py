import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch
from methods import backbone
import network_test
from methods import visiontransformer as vits



# --- conventional supervised training ---
class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, params, tf_path=None, loss_type='softmax'):
        super(BaselineTrain, self).__init__()

        # feature encoder
        self.feature = model_func()
        self.params = params

   
        self.feature.final_feat_dim = self.feature.num_features
     


        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)

        elif loss_type == 'dist':
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)

        self.loss_type = loss_type
        self.loss_fn = nn.CrossEntropyLoss()

        self.num_class = num_class
        self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None


    def forward_loss(self, x, y):
        # forward feature extractor
        x = x.cuda()
        x_fea = self.feature.forward(x, params=self.params)
        CLS_scores = self.classifier.forward(x_fea)

        y = y.cuda()
        loss_CLS = self.loss_fn(CLS_scores, y)
        return loss_CLS

    def train_loop(self, epoch,train_loader, optimizer, total_it):
        print_freq = len(train_loader) // 10
        avg_loss = 0


        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

 
            loss= self.forward_loss(x, y)
          
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()  # data[0]


            if (i + 1) % print_freq == 0:
               
                print('Epoch {:d} | Batch {:d}/{:d} | TotalLoss {:f}'.format(epoch, i + 1, len(train_loader),
                                                                             avg_loss / float(i + 1)))
            if (total_it + 1) % 10 == 0:
                self.tf_writer.add_scalar('loss', loss.item(), total_it + 1)
            total_it += 1

        return total_it







    def test_loop(self, val_loader, params, epoch): 
        # train and test together
        params.ckp_path = params.checkpoint_dir + '/' + 'last_model.tar'
        train_dataset = params.dataset
        acc_dict = {}
        acc_str = {}
        novel_accs = []
        for d in params.eval_datasets:
            if d == 'ave':
                continue

            params.dataset = d
            output = network_test.test_single_ckp(params)
            acc = float(output.split('Acc = ')[-1].split('%')[0])
            acc_dict[d] = acc
            acc_str[d] = output

            if d != 'miniImagenet':
                novel_accs.append(acc)

        acc_dict['ave'] = sum(novel_accs) / len(novel_accs)
        acc_str['ave'] = '%4.2f%%' % (acc_dict['ave'])

        params.dataset = train_dataset

        return acc_dict, acc_str


    def test_loop_orignial(self, val_loader):  
        # 只训不测
        return -1


