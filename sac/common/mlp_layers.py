import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from tensordict import from_modules 
from copy import deepcopy

class Ensamble(nn.Module):
    """
    An ensamble of MLP networks used primarily for the critic.
    
    """
    def __init__(self, modules, **kwargs):
        super().__init__()
        self.params = from_modules(*modules, as_module=True)
        with self.params[0].data.to("meta").to_module(modules[0]): #This I need to understand better 
            self.module = deepcopy(modules[0])
        self._repr = str(self.module[0])
        self._n = len(modules)

    def __len__(self):
        return self._n

    def _call(self, params, *args, **kwargs):
        with params.to_module(self.module):
            return self.module(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return  torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs) # This is vectorizing the call to the module, but I need to understand what the parameters are.and

    def __repr__(self):
        return f"Vectorized {len(self)}x {self._repr}"

class NormedLinear(nn.Linear):
    """
    Linear Layer that normalizes the input features to have unit variance
    before applying the linear transformation.
    This is useful for preventing exploding gradients in deep networks.
    """
    def __init__(self, *args, dropout=0., act=None, **kwargs):
        super().__init__(*args, **kwargs)
        sefl.ln = nn.LayerNorm(self.out_features)
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act 
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None 

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))
    
    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias is not None}{repr_dropout}, "\
            f"act={self.act.__class__.__name__})"

    def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
        """
        Standard MLP with layernorm mish activations and dropout.
        """
        if isinstance(mlp_dims, int):
            mlp_dims = [mlp_dims] 
        dims = [in_dim] + mlp_dims + [out_dim]
        mlp = nn.ModuleList()
        for i in range(len(dims)-2):
            mlp.append(
                NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0))
            )
        final_dropout = dropout if len(dims) == 2 else 0
        mlp.append(NormedLinear(dims[-2], dims[-1], dropout=final_dropout, act=act) if act else nn.Linear(dims[-2], dims[-1]))
        return mlp
        