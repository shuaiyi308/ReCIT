import os
import random
import numpy as np
import torch
import torch.optim
import time
from data.datamgr import SimpleDataManager, SetDataManager
from methods import backbone
from methods.backbone import model_dict
from methods.baselinetrain import BaselineTrain
from utils import load_state_to_the_backbone
from options import parse_args, get_resume_file, load_warmup_state
from methods.protonet import ProtoNet



def train(base_loader, val_loader, model, start_epoch, stop_epoch, params,labeled_target_loader=None):
  # get optimizer and checkpoint path
  if params.stage == 'pretrain':
    if params.optimizer == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), params.lr)
    elif params.optimizer == 'sgd':
      optimizer = torch.optim.SGD(model.parameters(), params.lr, momentum=0.9, nesterov=True, weight_decay=params.decay)
    elif params.optimizer == 'adamW':
      if 'VIT' in params.model:
        total_params = sum(p.numel() for p in model.parameters())
        #print(f"total params: {total_params:,}") 
        #for name, param in model.named_parameters():
          #print(f"{name}: {param.numel():,} parameters")
        
        scratch_params = []
        pretrain_params = []
        halfPret_params = []
        lr_small_params = []
        for n, p in model.named_parameters():  
          if 'feature' in n:
            
            if '!!!!' in n: 
              halfPret_params.append(p)
              #print('halfpretrain',n)
              continue
            elif '!!!' in n:
              lr_small_params.append(p)
              #print('small_lr',n)
              continue
                
            else:
              pretrain_params.append(p)
              #print('pretrain',n)
              continue
          scratch_params.append(p)
        
        
        
        
        optimizer = torch.optim.AdamW(
                        [{'params': pretrain_params, 'lr': params.lr * 0.0001},
                         {'params': lr_small_params, 'lr': params.lr * 0.0001},
                         {'params': halfPret_params, 'lr': params.lr * 0.0001},
                         {'params': scratch_params}
                         ],
                        lr=params.lr,
                        weight_decay=params.decay)
                
      else:
        optimizer = torch.optim.AdamW(model.parameters(), params.lr, weight_decay=params.decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)

  elif params.stage == 'metatrain': # not used
    if params.optimizer == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), params.lr)
    elif params.optimizer == 'sgd':
      optimizer = torch.optim.SGD(model.parameters(), params.lr, momentum=0.9, nesterov=True, weight_decay=params.decay)
    elif params.optimizer == 'adamW':
      optimizer = torch.optim.AdamW(model.parameters(), params.lr, weight_decay=params.decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)

  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # for validation
  max_acc={}
  for d in params.eval_datasets:
    max_acc[d] = 0

  total_it = 0
  with open('%s.txt' % (params.name), 'w', encoding='utf-8') as f1:
    f1.write('Testing the %s on %s~~~\n' % (params.name, params.eval_datasets))
  # start
 
  #start_time = time.time()
  for epoch in range(start_epoch,stop_epoch):
    model.train()
    params.epoch = epoch

    total_it = model.train_loop(epoch,base_loader, optimizer, total_it)

    model.eval()
    

    outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
    torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)       

    log_str = 'epoch: %d, '%epoch
    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
            acc,acc_str = model.test_loop(val_loader, params, epoch)
            with open('%s.txt' % (params.name), 'a', encoding='utf-8') as f1:
              f1.write('\nepoch%d:\n' % (epoch))
            for da, ac in acc_str.items():
              with open('%s.txt' % (params.name), 'a', encoding='utf-8') as f1:
                f1.write('%s:%s\t' % (da,ac))
              if max_acc[da]<acc[da]:
                max_acc[da]=acc[da]
                outfile = os.path.join(params.checkpoint_dir, 'best_%s_model.tar'%da)
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            with open('%s.txt' % (params.name), 'a', encoding='utf-8') as f1:
              f1.write('｜Highest\t')
              for da, ac in max_acc.items():
                f1.write('%s:%4.2f%%\t' % (da, ac))
        
    scheduler.step() 
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Time taken: {elapsed_time:.2f} seconds")
    


    
  
  with open('%s.txt' % (params.name), 'a', encoding='utf-8') as f1:
    f1.write('\nHighest accuracy:\n')
  for da, ac in max_acc.items():
    with open('%s.txt' % (params.name), 'a', encoding='utf-8') as f1:
      f1.write('%s:%4.2f%%\t' % (da, ac))
  


  return model


# --- main function ---
if __name__=='__main__':

  # set random seed
  seed = 0
  print("set seed = %d" % seed)
  random.seed(seed)
  os.environ["PYTHONSEED"] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) 
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False 
  torch.backends.cudnn.enabled = True 
  torch.manual_seed(seed)


  # parser argument
  params = parse_args('train')

  print('--- baseline training: {} ---\n'.format(params.name))
  print(params)


  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  #output/log/ params.name
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  # output/checkpoints/ params.name
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)


  # dataloader
  print('\n--- prepare source dataloader ---')
  source_base_file  = os.path.join(params.data_dir, 'miniImagenet', 'base.json')
  source_val_file   = os.path.join(params.data_dir, 'miniImagenet', 'val.json')

  # model
  print('\n--- build model ---')
  image_size = 224

  if params.stage == 'pretrain':
    print('  pre-training the model using only the miniImagenet source data')
    base_datamgr    = SimpleDataManager(image_size, batch_size=16)
    val_datamgr     = SimpleDataManager(image_size, batch_size=8)
    base_loader     = base_datamgr.get_data_loader(source_base_file , aug=params.train_aug )
    val_loader      = val_datamgr.get_data_loader(source_val_file, aug=False)

    model           = BaselineTrain(model_dict[params.model], params.num_classes,params, tf_path=params.tf_dir)  #add dis

  elif params.stage == 'metatrain': # not used
    log(out, '  meta training the model using the miniImagenet data and the {} auxiliary data'.format(params.target_set))

    #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    n_query = 15 #max(1, int(16* params.test_n_way/params.train_n_way))

    train_few_shot_params = dict(n_way = params.train_n_way, n_support = params.n_shot)
    base_datamgr = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(source_base_file, aug = params.train_aug )

    test_few_shot_params = dict(n_way = params.test_n_way, n_support = params.n_shot)
    val_datamgr = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader( source_val_file, aug = False)

    ##########################
    model = ProtoNet(model_dict[params.model], params.train_n_way, params.n_shot)
    

  else:
    raise ValueError('Unknown method')

  model = model.cuda()


  # load model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      tmp = torch.load(resume_file)
      start_epoch = tmp['epoch']+1
      model.load_state_dict(tmp['state'])
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  elif 'pretrain' not in params.stage:
    if params.warmup == 'gg3b0':
      raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
    
    state = torch.load(params.warmup)['state']
    print(state.keys())
    print('here')
    print(model.state_dict().keys())
    model.load_state_dict(state, False)

  # training
  print('\n--- start the training ---')

  model = train(base_loader, val_loader, model, start_epoch, stop_epoch, params)

