from fastai.basics import *

def iter_children(mod:nn.Module, base=''):
    '''Iterates through (name,module) pairs of named children recursively'''
    for n,ch in mod.named_children():
        if list(ch.children()): yield from iter_children(ch, base+n+'.')
        else:                   yield base+n,ch

class ErrorCallback(LearnerCallback):
    def __init__(self, lrn:Learner):
        super().__init__(lrn)
        self.err_loss,self.err_input,self.err_output = None,None,None
        
    def on_train_begin(self, **kwargs):
        def hook(mod, inps, outs):
            nfs = []
            for inp in inps:
                if inp is None: continue
                inp = inp.detach()
                nfs.append((
                    (inp == inp.new_full((1,), np.inf)).sum().cpu(), # Count non-finites
                    (inp == inp.new_full((1,), np.nan)).sum().cpu()  # On GPU so don't check yet
                ))
            return (mod, nfs)
        self.module_names = {m: n for n,m in iter_children(mdl_mish)}
        self.hooks = callbacks.Hooks([m for m in self.module_names.keys() if hasattr(m, 'weight')],
                                     hook, is_forward=False, detach=False)
        
    def on_batch_end(self, num_batch, last_loss, last_input, last_output, pbar, **kwargs):
        if not np.isfinite(last_loss) and self.err_loss is None:
            self.err_loss,self.err_input,self.err_output = last_loss,last_input,last_output
            pbar.write(f"Non-finite loss on batch {num_batch}")
            return {'stop_epoch': True, 'stop_training': True}
    
    def on_backward_end(self, num_batch, last_loss, last_input, last_output, pbar, **kwargs):
        for mod,nfs in self.hooks.stored:
            infs,nans = 0,0
            for inf,nan in nfs:
                infs += inf
                nans += nan
            if infs or nans:
                name = self.module_names[mod]
                pbar.write(f"Non-finite gradients on batch {num_batch} from child {name}, {infs} inf, {nans} nan. Aborting.")
                self.err_loss,self.err_input,self.err_output = last_loss,last_input,last_output
                return {'stop_epoch': True, 'stop_training': True}
            
    def on_train_end(self, **kwargs): self.hooks.remove()
    
    def on_epoch_end(self, **kwargs):
        if self.err_loss is not None: return {'stop_training': True}