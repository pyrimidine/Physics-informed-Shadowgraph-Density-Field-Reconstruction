import torch
import torch.nn.functional as F
import torch
from torch.nn import functional as F


def dfdx(f, h):
    # h=dx
    dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h 	
    dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
    dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
    dfdxi=torch.cat((dfdxi_left[:,:,:,0:2], dfdxi_internal, dfdxi_right[:,:,:,-2:]),3)
    return dfdxi

def dfdy(f, h):
    # h=dy
    dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h	
    dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
    dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
    dfdeta=torch.cat((dfdeta_low[:,:,0:2,:], dfdeta_internal, dfdeta_up[:,:,-2:,:]),2)
    return dfdeta



def d2fdx2_ghost(f, h):
    left_ghost = f[:, :, :, :1].repeat(1, 1, 1, 2)
    right_ghost = f[:, :, :, -1:].repeat(1, 1, 1, 2)

    f_padded = torch.cat([left_ghost, f, right_ghost], dim=3)

    dfdxi_internal = (-f_padded[:, :, :, :-4] + 16 * f_padded[:, :, :, 1:-3] - 30 * f_padded[:, :, :, 2:-2] 
                      + 16 * f_padded[:, :, :, 3:-1] - f_padded[:, :, :, 4:]) / (12 * h**2)
    
    return dfdxi_internal

def d2fdy2_ghost(f, h):
    up_ghost = f[:, :, :1, :].repeat(1, 1, 2, 1)
    down_ghost = f[:, :, -1:, :].repeat(1, 1, 2, 1)

    f_padded = torch.cat([up_ghost, f, down_ghost], dim=2)

    dfdxi_internal = (-f_padded[:, :, :-4, :] + 16 * f_padded[:, :, 1:-3, :] - 30 * f_padded[:, :, 2:-2, :] 
                      + 16 * f_padded[:, :, 3:-1, :] - f_padded[:, :, 4:, :]) / (12 * h**2)

    return dfdxi_internal

def d2fdx2(f, h):
    dfdxi_internal = (-f[:, :, :, :-4] + 16 * f[:, :, :, 1:-3] - 30 * f[:, :, :, 2:-2] 
                      + 16 * f[:, :, :, 3:-1] - f[:, :, :, 4:]) / (12 * h**2)
    
    dfdxi_left = (f[:, :, :, :2] - 2 * f[:, :, :, 1:3] + f[:, :, :, 2:4]) / h**2 
    dfdxi_right = (f[:, :, :, -2:] - 2 * f[:, :, :, -3:-1] + f[:, :, :, -4:-2]) / h**2  
    
    return torch.cat((dfdxi_left, dfdxi_internal, dfdxi_right), dim=3)

def d2fdy2(f, h):
    dfdxi_internal = (-f[:, :, :-4, :] + 16 * f[:, :, 1:-3, :] - 30 * f[:, :, 2:-2, :] 
                      + 16 * f[:, :, 3:-1, :] - f[:, :, 4:, :]) / (12 * h**2)
    
    dfdxi_up = (f[:, :, :2, :] - 2 * f[:, :, 1:3, :] + f[:, :, 2:4, :]) / h**2  
    dfdxi_low = (f[:, :, -2:, :] - 2 * f[:, :, -3:-1, :] + f[:, :, -4:-2, :]) / h**2  
       
    return torch.cat((dfdxi_up, dfdxi_internal, dfdxi_low), dim=2)



def possion_eq(Is, scale_log_n, I0, X_len=88.47*1e-3, Y_len=109.83*1e-3, L=1.3+0.78, D=0.0015, depth=None): # (b, c, h, w)

    I0 = I0.repeat(Is.shape[0], 1, 1, 1)
    dx = X_len/Is.shape[2] 
    dy = Y_len/Is.shape[3]
    log_n = 1e-3 * scale_log_n 


    if depth is None:
        left_side = - (Is - I0) / ((I0 + 1e-4) * (L * D))
    else:
        left_side = - (Is - I0) / ((I0 + 1e-4) * (L * depth))


    right_side = d2fdx2_ghost(log_n, dx) + d2fdy2_ghost(log_n, dy)

    left_side = torch.sigmoid(left_side)
    right_side = torch.sigmoid(right_side)

    return left_side, right_side
