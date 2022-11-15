from turtle import pd
from torch.quantization import FakeQuantizeBase 
import torch
from torch.quantization.observer import MinMaxObserver


def _is_float_qparams(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_channel_affine_float_qparams, ]


def _is_per_channel(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_channel_symmetric, torch.per_channel_affine, torch.per_channel_affine_float_qparams]

def _is_per_tensor(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]

def _is_symmetric_quant(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]

class ObsOps(torch.autograd.Function):


    @staticmethod
    def forward(ctx, X, clip_hi, clip_lo, zero_point, activation_post_process, observer_enabled, fake_quant_enabled,
                    setup, n_levels, quant_min, quant_max, device, qscheme):
        # ctx is a context object that can be used to stash information
        # for backward computation

        

        if not setup:
                activation_post_process(X)
                scale, zero_point = activation_post_process.calculate_qparams()
                zero_point = -zero_point + quant_min
                clip_hi = ((zero_point + ((quant_max-quant_min) -1))*scale).data
                clip_lo = (zero_point*scale).data
            
                where_x_hi = (clip_hi <= X)
                where_x_nc = (clip_lo <= X) * (X < clip_hi)
                # if observer_enabled[0] == 1:
                #     activation_post_process(X)
                #     scale, zero_point = activation_post_process.calculate_qparams()
                #     clip_hi = ((zero_point + (n_levels -1))*scale).data
                #     clip_lo = (zero_point*scale).data
                    
                #     scale, zero_point = scale.to(scale.device), zero_point.to(zero_point.device)
                    
                # else: 
                #     scale, zero_point = activation_post_process.calculate_qparams()
                #     clip_hi = ((zero_point + (n_levels -1))*scale).data
                #     clip_lo = (zero_point*scale).data
                    
                #     scale, zero_point = scale.to(scale.device), zero_point.to(zero_point.device)
                    

                # if fake_quant_enabled[0] == 1:
                    
                #     X = torch.fake_quantize_per_tensor_affine(
                #              X, scale.data.item(), int(zero_point.data.item()),
                #              quant_min, quant_max)
 
                ctx.save_for_backward(where_x_hi, where_x_nc, clip_hi)
               
                return X, scale, zero_point, clip_hi, clip_lo
 
 
        else:
            where_x_hi = (clip_hi <= X)
            where_x_nc = (clip_lo <= X) * (X < clip_hi)
            activation_post_process._validate_qmin_qmax(quant_min, quant_max)
            
            scale = torch.ones(activation_post_process.min_val.size(), dtype=torch.float32, device=device)
            zero_point = torch.zeros(activation_post_process.min_val.size(), dtype=torch.int64, device=device)
            scale = torch.max(zero_point, clip_hi)/(float(quant_max - quant_min) / 2)

            scale = (clip_hi - clip_lo) / float(quant_max - quant_min)
            zero_point = - torch.round(clip_lo / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

            # For scalar values, cast them to Tensors of size 1 to keep the shape
            # consistent with default values in FakeQuantize.
            if len(scale.shape) == 0:
                # TODO: switch to scale.item() after adding JIT support
                scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
            if len(zero_point.shape) == 0:
                # TODO: switch to zero_point.item() after adding JIT support
                zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype, device=device)
                if qscheme == torch.per_channel_affine_float_qparams:
                    zero_point = torch.tensor([float(zero_point)], dtype=zero_point.dtype, device=device)

            if observer_enabled[0] == 1:
                activation_post_process(X)
                
                scale, zero_point = scale.to(scale.device), zero_point.to(zero_point.device)
                if scale.shape != scale.shape:
                    scale.resize_(scale.shape)
                    zero_point.resize_(zero_point.shape)
            
                

            if fake_quant_enabled[0] == 1:
                
                X = torch.fake_quantize_per_tensor_affine(
                        X, scale.data.item(), int(zero_point.data.item()),
                        quant_min, quant_max)

            ctx.save_for_backward(where_x_hi, where_x_nc, clip_hi)
            
            
            return X, scale, zero_point, clip_hi, clip_lo


    @staticmethod
    def backward(ctx, grad_output, grad_scale, grad_zero_point, grad_clip_hi, grad_clip_lo):
        
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        #import pdb; pdb.set_trace()
        
        where_x_hi, where_x_nc, clip_hi = ctx.saved_tensors 
        zero = torch.zeros(1).to(where_x_hi.device)
        g_out = torch.where(where_x_nc, grad_output, zero)
        g_clip_hi = torch.where(where_x_hi, grad_output, zero).sum().reshape(clip_hi.shape)
        
        
        return g_out, g_clip_hi, None, None, None, None, None, None, None, None, None, None, None


class PACTFakeQuantize(FakeQuantizeBase):
    r""" Simulate the quantize and dequantize operations in training time.
    The output of this module is given by::
        x_out = (
          clamp(round(x/scale + zero_point), quant_min, quant_max) - zero_point
        ) * scale
    * :attr:`scale` defines the scale factor used for quantization.
    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to
    * :attr:`fake_quant_enabled` controls the application of fake quantization on tensors, note that
      statistics can still be updated.
    * :attr:`observer_enabled` controls statistics collection on tensors
    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
        allowable values are torch.qint8 and torch.quint8.
    Args:
        observer (module): Module for observing statistics on input tensors and calculating scale
          and zero-point.
        observer_kwargs (optional): Arguments for the observer module
    Attributes:
        activation_post_process (Module): User provided module that collects statistics on the input tensor and
          provides a method to calculate scale and zero-point.
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, observer = MinMaxObserver, quant_min=None, quant_max=None, learn = False, **observer_kwargs):
        super().__init__()
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            assert quant_min <= quant_max, \
                'quant_min must be less than or equal to quant_max'
        dtype = observer_kwargs.get("dtype", torch.quint8)
        assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
        assert quant_max <= torch.iinfo(dtype).max, 'quant_max out of bound'
        observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        
        self.activation_post_process = observer(**observer_kwargs)
        
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=zero_point_dtype))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        assert _is_per_channel(self.qscheme) or \
            _is_per_tensor(self.qscheme), \
            'Only per channel and per tensor quantization are supported in fake quantize' + \
            ' got qscheme: ' + str(self.qscheme)
        self.is_per_channel = _is_per_channel(self.qscheme)
        self.learn = learn 
        if self.learn:
            self.register_buffer('clip_lo', torch.tensor([0.0], dtype=torch.float))
            self.register_parameter('clip_hi', torch.nn.Parameter(torch.tensor([255.0], dtype = torch.float, requires_grad=True)))
            self.n_levels = 256
            self.setup = False  
            self.device = observer_kwargs.get("device", 'cpu')
        
    
    

    @torch.jit.export
    def calculate_qparams(self):    
        
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
      
       
        if self.learn:

            X, scale, zero_point, clip_hi, clip_lo = ObsOps.apply(X, self.clip_hi, self.clip_lo, self.zero_point, self.activation_post_process, 
            self.observer_enabled, self.fake_quant_enabled, self.setup, self.n_levels, self.quant_min, self.quant_max, self.device, self.qscheme)
            self.scale.data = scale.data
            self.zero_point.data = zero_point.data
            self.clip_lo.data = clip_lo.data
            self.clip_hi.data = clip_hi.data

            return X

        else:
            if self.observer_enabled[0] == 1:
                self.activation_post_process(X)
                _scale, _zero_point = self.calculate_qparams()
                _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
                if self.scale.shape != _scale.shape:
                    self.scale.resize_(_scale.shape)
                    self.zero_point.resize_(_zero_point.shape)
                self.scale.copy_(_scale)
                self.zero_point.copy_(_zero_point)

            if self.fake_quant_enabled[0] == 1:
                if self.is_per_channel:
                    X = torch.fake_quantize_per_channel_affine(
                        X, self.scale, self.zero_point,
                        self.ch_axis, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
                else:
                    X = torch.fake_quantize_per_tensor_affine(
                        X, self.scale, self.zero_point,
                        self.activation_post_process.quant_min, self.activation_post_process.quant_max)
            return X

            


        
        

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.activation_post_process.quant_min, self.activation_post_process.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(PACTFakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'scale':
                    self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super(PACTFakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                        missing_keys, unexpected_keys, error_msgs)