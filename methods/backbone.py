

import torch

from methods import visiontransformer as vits






# --- VIT with dino ---
def vit_small_dino(**kwargs):
    model = vits.vit_small()
        
    state_dict = torch.load('methods/dino_deitsmall16_pretrain.pth')

    #model.load_state_dict(state_dict, strict=True)
    model_state_dict = model.state_dict()
    num_loaded_params = 0
    for name in model_state_dict.keys():

            #print(name, model_state_dict[name].shape)


      if name in state_dict.keys():
          model_state_dict[name] = state_dict[name]
          num_loaded_params += 1


    model.load_state_dict(model_state_dict, strict=True)
    #print(num_loaded_params, 'params loaded')



    return model





model_dict = dict(VIT_S = vit_small_dino)
