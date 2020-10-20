import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import fft_utils
from matplotlib import pyplot as plt
import fastmri
from fastmri.data import transforms as T

import cv2

class Net(nn.Module):
	# Here we define the Neural Network Structure
    def __init__(self, ksize=5, n_cascade=2):
        super(Net, self).__init__()
        
        self.shuffle_down_4 = ComplexShuffleDown(4)
        self.shuffle_up_4 = ComplexShuffleUp(4)
        self.shuffle_up_2 = ComplexShuffleUp(2)
        self.convBlock1 = ConvBlock1(16, 64)

        # After the 4x Downsampling we have to split the computation in 3 branches
        # Each will execute a different operation that will result in combining the output images with the
        # laplacian decomposition results

        # Each of the followings is the ConvBlock2, we needed to change the input/output channels
        # in order to align to the paper specifics
        self.branch1 = nn.Sequential(ComplexConv2d(4, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, ksize ** 2, 3, 1, 1),
                                 nn.ReLU())
        self.branch2 = nn.Sequential(ComplexConv2d(16, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, ksize ** 2, 3, 1, 1),
                                 nn.ReLU())
        self.branch3 = nn.Sequential(ComplexConv2d(64, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, 64, 3, 1, 1),
                                 nn.ReLU(),
                                 ComplexConv2d(64, ksize ** 2, 3, 1, 1),
                                 nn.ReLU())

        #Per-pixel convolution operation
        self.pixel_conv = PerPixelConv()
        
        self.pyramid_decompose = PyramidDecompose()
        self.pyramid_reconstruct = PyramidReconstruct()

        self.n_cascade = n_cascade

    def forward(self, mr_img, mk_space, mask):
        # Loop over num = cascades number
        for i in range(self.n_cascade):
            # The laplacian decomposition will produce different images composing the gaussian/laplacian pyramids
            # The gaussians are the downsampled images from the original one
            # The laplacian represents the 'errors' within the images related to the gaussian images
            #gaussian_pyramid, laplacian_pyramid = self.pyramid_decompose(mr_img)
			#
            # We need the first two bigger images
            #lap_1, lap_2 = laplacian_pyramid[0], laplacian_pyramid[1]
            # We need the smallest gaussian image
            #gaussian_3 = gaussian_pyramid[-1]
			
            pip = PIP()
            gaussian_3, lap_1, lap_2 = pip(mr_img)
			
            # Resize along with the input/output channels
            mr_img = self.shuffle_down_4(mr_img)

            # The convBlock1 is here performed to extract shallow features
            mr_img = self.convBlock1(mr_img)
            
            # ---- 3 branches ----
            # 4x
            branch_1 = self.shuffle_up_4(mr_img)
            branch_1 = self.branch1(branch_1)
            # 2x
            branch_2 = self.shuffle_up_2(mr_img)
            branch_2 = self.branch2(branch_2)
            # 1x
            branch_3 = self.branch3(mr_img)

            output1 = torch.stack((self.pixel_conv(branch_1[...,0], lap_1[...,0]),
                                   self.pixel_conv(branch_1[...,1], lap_1[...,1])), dim=-1)
            output2 = torch.stack((self.pixel_conv(branch_2[...,0], lap_2[...,0]),
                                   self.pixel_conv(branch_2[...,1], lap_2[...,1])), dim=-1)
            output3 = torch.stack((self.pixel_conv(branch_3[...,0], gaussian_3[...,0]),
                                   self.pixel_conv(branch_3[...,1], gaussian_3[...,1])), dim=-1)
			
			# Performing the 2x linear upsample in order to add the different branches correctly
            output = self.pyramid_reconstruct(output3, output2)
            output = self.pyramid_reconstruct(output, output1)
            
            # The DataConsistency Layer step is done here
            mr_img = fft_utils.ifft2((1.0 - mask) * fft_utils.fft2(output) + mk_space)
        return mr_img

class PerPixelConv(nn.Module):
    def __init__(self):
        super(PerPixelConv, self).__init__()

    def forward(self, kernel, image):
        batch, ksize2, height, width = kernel.size()
        ksize = np.int(np.sqrt(ksize2))
        padding = (ksize - 1) // 2
        # Before computing the per pixel convolution step we need to prepare the
        # inputs by doing some operation like padding and reshape in order to compute
        # the multiplication between matrices
        image = F.pad(image, (padding, padding, padding, padding))
        image = image.unfold(2, ksize, 1).unfold(3, ksize, 1)
        image = image.permute(0, 2, 3, 1, 5, 4).contiguous()
        image = image.reshape(batch, height, width, 1, -1)
        kernel = kernel.permute(0, 2, 3, 1).unsqueeze(-1)
        output = torch.matmul(image, kernel)
        output = output.reshape(batch, height, width, -1)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output

	# The residual-block is used in the ConvBlock1 loop
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.residualblock = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return x + self.residualblock(x)


class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(ConvBlock1, self).__init__()
        convBlock1 = [ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU()]
        # Due to performance issues, we needed to reduce the conv block cycle steps from 16 to 2
        for i in range(2):
            convBlock1.append(ResidualBlock(out_channels,out_channels))
        convBlock1.append(ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1))
        convBlock1.append(nn.ReLU())
        self.convBlock1 = nn.Sequential(*convBlock1)

    def forward(self, x):
        return self.convBlock1(x)

class PIP(nn.Module):
	def __init__(self):
		super(PIP, self).__init__()
		
	def forward(self, x):
		# generate Gaussian pyramid for A
		img = x.squeeze(0).squeeze(0)
		#img =  fastmri.complex_abs(img)
		print(img.shape)
		gpA = [img]
		for i in range(3):
			real = img[..., 0]
			print(img.size())
			imm = img[..., 1]
			Gr = cv2.pyrDown(real)
			Gi = cv2.pyrDown(imm)
			img = torch.stack(Gr, Gi)
			gpA.append(img)

		# generate Laplacian Pyramid for A
		
		lpA = []
		for i in range(3,0,-1):
			size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])
			g = gpA[i-1]
			GEr = cv2.pyrUp(g[...,0],  dstsize=size)
			GEi = cv2.pyrUp(g[...,1],  dstsize=size)
			GE = torch.stack(GEr, GEi)
			L = cv2.subtract(g, GE)
			lpA.append(L)
		return gpA[2].unsqueeze(0).unsqueeze(0), lpA[2].unsqueeze(0).unsqueeze(0), lpA[1].unsqueeze(0).unsqueeze(0)

class PyramidDecompose(nn.Module):
    # We take the input image and we decompose it in order to have the 2 gaussian images and a
    # laplacian one. This will be useful later when performing the per-pixel convolution step
    def __init__(self):
        super(PyramidDecompose, self).__init__()
        kernel = np.float32([1, 4, 6, 4, 1])
        # Outer product between kernel
        kernel = np.outer(kernel, kernel)
        kernel = kernel[:, :, None, None] / kernel.sum()
        kernel = torch.from_numpy(np.transpose(kernel, (2, 3, 0, 1)))
        self.downfilter = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.downfilter.weight = nn.Parameter(kernel, requires_grad=False)
        self.upfilter = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.upfilter.weight = nn.Parameter(kernel * 4, requires_grad=False)

    @staticmethod
    def downsample(x):
            return x[:, :, ::2, ::2]

    @staticmethod
    def upsample(x):
        batch, channel, height_in, width_in = x.size()
        height_out, width_out = height_in * 2, width_in * 2
        x_upsample = torch.zeros((batch, channel, height_out, width_out))#, device='cuda')
        x_upsample[:, :, ::2, ::2] = x
        return x_upsample

    def forward(self, x):
        # The input image of the gaussian matrix is already given, since we use the input image
        gaussian_pyramid = [x]
        laplacian_pyramid = []
        # We want 2 images, so the reduction step will be done two times
        for i in range(2):
            downsample_result = torch.stack((self.downsample(self.downfilter(gaussian_pyramid[-1][..., 0])),
                                             self.downsample(self.downfilter(gaussian_pyramid[-1][..., 1]))), dim=-1)
            gaussian_pyramid.append(downsample_result)
            upsample_result = torch.stack((self.upfilter(self.upsample(downsample_result[..., 0])),
                                           self.upfilter(self.upsample(downsample_result[..., 1]))), dim=-1)
            # Residual is the first residual of the 'laplacian' image
            residual = gaussian_pyramid[-2] - upsample_result
            laplacian_pyramid.append(residual)
        return gaussian_pyramid, laplacian_pyramid


class PyramidReconstruct(nn.Module):
    def __init__(self):
        super(PyramidReconstruct, self).__init__()
        kernel = np.float32([1, 4, 6, 4, 1])
        kernel = np.outer(kernel, kernel)
        kernel = kernel[:, :, None, None] / kernel.sum()
        kernel = torch.from_numpy(np.transpose(kernel, (2, 3, 0, 1)))
        self.upfilter = nn.Conv2d(1, 1, 5, 1, 2, bias=False)
        self.upfilter.weight = nn.Parameter(kernel * 4, requires_grad=False)

    @staticmethod
    def upsample(x):
        batch, channel, height_in, width_in = x.size()
        height_out, width_out = height_in * 2, width_in * 2
        x_upsample = torch.zeros((batch, channel, height_out, width_out))#, device='cuda')
        x_upsample[:, :, ::2, ::2] = x
        return x_upsample

    def forward(self, x_gaussian, x_laplacian):
        upsample_result = torch.stack((self.upfilter(self.upsample(x_gaussian[..., 0])), self.upfilter(self.upsample(x_gaussian[..., 1]))), dim=-1)
        recon = upsample_result + x_laplacian
        return recon


class ComplexShuffleDown(nn.Module):
    # Perform resize
    def __init__(self, factor):
        super(ComplexShuffleDown, self).__init__()
        self.factor = factor

    def forward(self, x):
        batch, channel_in, height_in, width_in, complex_channel = x.size()
        channel_out = channel_in * (self.factor ** 2)
        height_out = height_in // self.factor
        width_out = width_in // self.factor
        output = x.view(batch, channel_in, height_out, self.factor, width_out, self.factor, complex_channel)
        output = output.permute(0, 1, 5, 3, 2, 4, 6).contiguous()
        output = output.view(batch, channel_out, height_out, width_out, complex_channel)
        return output


class ComplexShuffleUp(nn.Module):
    # Perform resize
    def __init__(self, factor):
        super(ComplexShuffleUp, self).__init__()
        self.factor = factor

    def forward(self, x):
        batch, channel_in, height_in, width_in, complex_channel = x.size()
        channel_out = channel_in // (self.factor ** 2)
        height_out = height_in * self.factor
        width_out = width_in * self.factor
        output = x.view(batch, channel_out, self.factor, self.factor, height_in, width_in, complex_channel)
        output = output.permute(0, 1, 4, 3, 5, 2, 6).contiguous()
        output = output.view(batch, channel_out, height_out, width_out, complex_channel)
        return output

class ComplexConv2d(nn.Module):
    # In order to compute the convolution step on complex data that required an additive dimension
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()

        self.conv_real = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_imag = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        real = self.conv_real(x[..., 0]) - self.conv_imag(x[..., 1])
        imag = self.conv_imag(x[..., 0]) + self.conv_real(x[..., 1])
        output = torch.stack((real, imag), dim=4)
        return output
