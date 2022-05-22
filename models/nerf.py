import torch
from itertools import chain
from hypnettorch.mnets import MLP
from hypnettorch.hnets import ChunkedHMLP, HMLP

class NeRF(torch.nn.Module):

    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, use_viewdirs=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs
       
        self.before_skip = MLP(n_in=self.input_ch, 
            n_out=self.W, hidden_layers=[self.W]*(self.D//2),
            no_weights=False, out_fn=torch.nn.ReLU())

        self.after_skip = MLP(n_in=self.input_ch + self.W, 
            n_out=self.W, hidden_layers=[self.W]*(self.D//2),
            no_weights=False, out_fn=torch.nn.ReLU()) 

        #assume we use viewdirs
        if use_viewdirs:
            self.out_sigma = MLP(n_in=self.W, 
                n_out=1, hidden_layers=[],
                no_weights=False)
            self.out_feature = MLP(n_in=self.W, 
                n_out=self.W, hidden_layers=[],
                no_weights=False, activation_fn=None)
            self.out_rgb = MLP(n_in=self.W + self.input_ch_views, 
                n_out=3, hidden_layers=[self.W//2],
                no_weights=False)

            self.internal_params = chain(self.before_skip.internal_params, self.after_skip.internal_params, self.out_sigma.internal_params, self.out_feature.internal_params, self.out_rgb.internal_params)
        else:
            self.output_linear = MLP(n_in=self.W, 
                n_out=4, hidden_layers=[],
                no_weights=False, activation_fn=None)

            self.internal_params = chain(self.before_skip.internal_params, self.after_skip.internal_params, self.output_linear.internal_params)

        

    def forward(self, x, weights=None):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        if self.use_viewdirs:
            bs_l, as_l, os_l, of_l, orgb_l = self.unpack_weights(weights)
        else:
            bs_l, as_l, ol_l = self.unpack_weights(weights)

        h, _=self.before_skip(h, weights=bs_l)
        h = torch.cat([h, input_pts], -1)
        h, _=self.after_skip(h,weights=as_l)

        if self.use_viewdirs:
            sigma = self.out_sigma(h,weights=os_l)
            feature = self.out_feature(h,weights=of_l)
            h = torch.cat([feature, input_views], -1)
            rgb = self.out_rgb(h, weights=orgb_l)
            return torch.cat([rgb, sigma], -1)
        else:
            return self.output_linear(h, weights=ol_l)


    @property
    def param_shapes(self) -> list:
        if self.use_viewdirs:
            return list(chain(self.before_skip.param_shapes, self.after_skip.param_shapes, self.out_sigma.param_shapes, 
                                self.out_feature.param_shapes, self.out_rgb.param_shapes))
        else:
            return list(chain(self.before_skip.param_shapes, self.after_skip.param_shapes, self.output_linear.param_shapes))

    def unpack_weights(self, weights) -> list:
        if(weights is None):
            print("No weights!")
            return None
        weights = weights.copy()
        bs_weights = []
        for param in self.before_skip.param_shapes:
            bs_weights.append(weights.pop(0))

        as_weights = []
        for param in self.after_skip.param_shapes:
            as_weights.append(weights.pop(0))

        if self.use_viewdirs:
            os_weights = []
            for param in self.out_sigma.param_shapes:
                os_weights.append(weights.pop(0))

            of_weights = []
            for param in self.out_feature.param_shapes:
                of_weights.append(weights.pop(0))

            orgb_weights = []
            for param in self.out_rgb.param_shapes:
                orgb_weights.append(weights.pop(0))

            assert len(weights)==0

            return bs_weights, as_weights, os_weights, of_weights, orgb_weights
        else:
            ol_weights = []
            for param in self.output_linear.param_shapes:
                ol_weights.append(weights.pop(0))
            
            assert len(weights)==0

            return bs_weights, as_weights, ol_weights