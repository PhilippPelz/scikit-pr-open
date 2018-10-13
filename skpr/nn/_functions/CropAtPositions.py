import torch as th
from torch.autograd import Function, Variable

import skpr.inout  as io


class CropAtPositions(Function):
    @staticmethod
    def forward(ctx, input, positions, gradient_scaling, size):
        io.logger.debug('CropAtPositions forward 1')

        ctx.input_size = input.size()
        ctx.size = size
        ctx.gradient_scaling = gradient_scaling
        ctx.save_for_backward(positions)

        # print 'CropAtPositions.forward's
        s = list((positions.size()[0],) + ctx.input_size)
        s[input.ndimension()] = ctx.size[1]
        s[input.ndimension() - 1] = ctx.size[0]
        S = th.Size(s)

        out = input.new().resize_(*S)

        # many streams may speed it up
        for i, pos in enumerate(positions):
            # print input.size()
            # print pos[0], pos[0]+size[0], pos[1],pos[1]+size[1]
            out[i].copy_(input[..., pos[0]:pos[0] + size[0], pos[1]:pos[1] + size[1]])
        io.logger.debug('CropAtPositions forward 2')
        return out

    @staticmethod
    def backward(ctx, grad_output):
        io.logger.debug('CropAtPositions backward 1')

        positions, = ctx.saved_tensors
        gradient_scaling = ctx.gradient_scaling
        grad_input = grad_output.data.new().resize_(*ctx.input_size)
        grad_input.zero_()
        i = 0
        for pos in positions:
            #            print pos[0], pos[0]+ctx.size[0], pos[1],pos[1]+ctx.size[1]
            #            print '22', grad_input[:, :, pos[0]:pos[0] + ctx.size[0], pos[1]:pos[1] + ctx.size[1]].size(), grad_output[i].data.size(), gradient_scaling.size()
            grad_input[:, :, pos[0]:pos[0] + ctx.size[0], pos[1]:pos[1] + ctx.size[1]] += grad_output[
                                                                                              i].data * gradient_scaling
            i += 1
        # p.var['dO_unscaled'] = grad_input[0, 0].cpu().numpy()
        # gabs = th.cuda.FloatTensor(*grad_input.size())
        # grad_input.abs(out=gabs)
        # grad_input.mul_(gradient_scaling)
        #        p.var['dO'] = grad_input[0, 0].cpu().numpy()
        grad_input = Variable(grad_input)
        io.logger.debug('CropAtPositions backward 2')
        return grad_input, None, None, None
