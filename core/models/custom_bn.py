"""
https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
"""

import torch 
from torch import nn

class MyNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, contineous=False, norm_lambda=0.1):
        super(MyNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
        self.contineous=contineous
        self.norm_lambda = norm_lambda
        self.temp_init = True

        
    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            # TODO: consider instance norm
            
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3]) #, unbiased=False)
            
            
            with torch.no_grad():
                
                if self.contineous:
                    self.running_mean = self.norm_lambda * mean + (1 - self.norm_lambda) * self.running_mean
                    self.running_var = self.norm_lambda * var  + (1 - self.norm_lambda) * self.running_var
                    mean = self.running_mean
                    var = self.running_var 
                    
                else:

                    mean = self.norm_lambda * mean + (1 - self.norm_lambda) * self.running_mean
                    var = self.norm_lambda * var + (1 - self.norm_lambda) * self.running_var    
        
        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input



def replace_layers(model, norm_lambda, contineous ):
    for n, module in model.named_children():

        if len(list(module.children())) > 0:
            replace_layers(module, norm_lambda, contineous)
            
        if isinstance(module, nn.BatchNorm2d):
            mybatch_norm = MyNorm2d(module.num_features, norm_lambda=norm_lambda, contineous=contineous)
            try:
                n = int(n)
                model[n] = mybatch_norm
            except:
                setattr(model, n, mybatch_norm)
            
def configure_adabn(model):
    for n, module in model.named_children():
        
        if len(list(module.children())) > 0:
            configure_adabn(module)
            
        if isinstance(module, nn.BatchNorm2d):
            module.training = True
            # To force the model to use batch statistics
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
                

class InstanceNormUpdates(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, contineous=False, norm_lambda=0.1):
        super(InstanceNormUpdates, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
        self.contineous=contineous
        self.norm_lambda = norm_lambda
        self.temp_init = True

        
    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            # TODO: consider instance norm
            mean = self.running_mean
            var = self.running_var            
        
        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        mean = input.mean([0, 2, 3])
            # use biased var in train
        var = input.var([0, 2, 3]) #, unbiased=False)
        
        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input



def replace_layers_instance(model):
    for n, module in model.named_children():

        if len(list(module.children())) > 0:
            replace_layers_instance(module)
            
        if isinstance(module, nn.BatchNorm2d):
            mybatch_norm = InstanceNormUpdates(module.num_features)
            try:
                n = int(n)
                model[n] = mybatch_norm
            except:
                setattr(model, n, mybatch_norm)
                
class Momentum(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, N=32, n=1):
        super(Momentum, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
        self.N=N
        self.n = n
        self.temp_init = True

        
    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            # TODO: consider instance norm
            
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3]) #, unbiased=False)
            
            with torch.no_grad():
                mean = (self.n/(self.n+self.N)) * mean + (self.N/(self.n+self.N)) * self.running_mean
                var = (self.n/(self.n+self.N)) * var + (self.N/(self.n+self.N)) * self.running_var 
                        
                input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
            
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


def replace_layers_momentum(model):
    for n, module in model.named_children():

        if len(list(module.children())) > 0:
            replace_layers_momentum(module)
            
        if isinstance(module, nn.BatchNorm2d):
            mybatch_norm = Momentum(module.num_features)
            try:
                n = int(n)
                model[n] = mybatch_norm
            except:
                setattr(model, n, mybatch_norm)