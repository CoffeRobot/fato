/*****************************************************************************/
/*  Copyright (c) 2016, Karl Pauwels, Alessandro Pieropan                    */
/*  All rights reserved.                                                     */
/*                                                                           */
/*  Redistribution and use in source and binary forms, with or without       */
/*  modification, are permitted provided that the following conditions       */
/*  are met:                                                                 */
/*                                                                           */
/*  1. Redistributions of source code must retain the above copyright        */
/*  notice, this list of conditions and the following disclaimer.            */
/*                                                                           */
/*  2. Redistributions in binary form must reproduce the above copyright     */
/*  notice, this list of conditions and the following disclaimer in the      */
/*  documentation and/or other materials provided with the distribution.     */
/*                                                                           */
/*  3. Neither the name of the copyright holder nor the names of its         */
/*  contributors may be used to endorse or promote products derived from     */
/*  this software without specific prior written permission.                 */
/*                                                                           */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      */
/*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        */
/*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    */
/*  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     */
/*  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   */
/*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         */
/*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    */
/*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    */
/*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     */
/*****************************************************************************/

//#define UNROLL_INNER
//#define IMUL(a, b) __mul24(a, b)
#include "gl_interop.h"
#include <iostream>
#include <sstream>

namespace fato{
namespace gpu {

texture<float, cudaTextureType2D, cudaReadModeElementType> d_float_texture;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> d_rgba_texture;

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

__device__ static float rgbaToGray(uchar4 rgba) {
  return (0.299f * (float)rgba.x + 0.587f * (float)rgba.y +
          0.114f * (float)rgba.z);
}

/*****************************************************************************/
/*                              KERNELS                                      */
/*****************************************************************************/
__global__ void copyTextureToFloat(float *out_image, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  // opengl and cuda have inverted image coordinate systems
  const int inv_y = height - 1;

  if (x < width && y < height) {
    float val = tex2D(d_float_texture, (float)x + 0.5f, (float)(y) + 0.5f);
    out_image[y * width + x] = val;
  }
}

__global__ void copyTextureToRGBA(uchar4 *out_image, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  // opengl and cuda have inverted image coordinate systems
  const int inv_y = height - 1;

  if (x < width && y < height) {
    out_image[y * width + x] =
        tex2D(d_rgba_texture, (float)x + 0.5f, (float)(inv_y - y) + 0.5f);
  }
}

__global__ void convertRGBAArrayToGrayVX_kernel(uchar *out_image, int width,
                                                 int height, int step) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const int inv_y = height - 1;
    uchar4 rgba = tex2D(d_rgba_texture, (float)x + 0.5f, (float)(inv_y - y) + 0.5f);

    float val = 0.299f * (float)rgba.x + 0.587f * (float)rgba.y +
              0.114f * (float)rgba.y;

    uchar *dst_row = (uchar *)(out_image + y * step);

    dst_row[x] = (uchar)val;
  }
}

/*****************************************************************************/
/*                              CALLING FUNCTIONS                            */
/*****************************************************************************/

std::runtime_error cudaException(const char *file, int line,
                                 cudaError_t error) {
  std::stringstream message;
  message << file << "," << line << ": "
          << std::string(cudaGetErrorString(error));
  return (std::runtime_error(message.str()));
}

void downloadTextureToRGBA(uchar4 *d_out_image, cudaArray *in_array, int width,
                           int height) {
  // Bind textures to arrays
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
  cudaError_t err =
      cudaBindTextureToArray(d_rgba_texture, in_array, channelDesc);

  if (err != cudaSuccess) {
    cudaException(__FILE__, __LINE__, err);
    std::cout << "downloadTextureToRGBA(102) :" +
                     std::string(cudaGetErrorString(err)) << std::endl;
    // exit(0);
  }

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);
  // std::cout << "calling the kernel " << std::endl;
  copyTextureToRGBA << <dimGrid, dimBlock>>> (d_out_image, width, height);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaException(__FILE__, __LINE__, err);
    std::cout << "downloadTextureToRGBA(114) :" +
                     std::string(cudaGetErrorString(err)) << std::endl;
    // exit(0);
  }

  cudaUnbindTexture(d_rgba_texture);
}

void downloadDepthTexture(float *d_out_image, cudaArray *in_array, int width,
                          int height) {
  // Bind textures to arrays
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  //cudaChannelFormatDesc channelDesc =
  //cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
  cudaError_t err =
      cudaBindTextureToArray(d_float_texture, in_array, channelDesc);
  if (err != cudaSuccess) {
    throw cudaException(__FILE__, __LINE__, err);
  }

  dim3 dimBlock(16, 8, 1);
  dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

  copyTextureToFloat << <dimGrid, dimBlock>>> (d_out_image, width, height);

  cudaUnbindTexture(d_float_texture);
}

void convertRGBArrayToGrayVX(uchar *d_out_image, cudaArray *in_array,
                             int width, int height, int step)
{
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaError_t err = cudaBindTextureToArray(d_rgba_texture, in_array, channelDesc);
    if (err != cudaSuccess) {
      throw cudaException(__FILE__, __LINE__, err);
    }

    dim3 dimBlock(16, 8, 1);
    dim3 dimGrid(iDivUp(width, dimBlock.x), iDivUp(height, dimBlock.y), 1);

    convertRGBAArrayToGrayVX_kernel << <dimGrid, dimBlock>>>
        (d_out_image, width, height, step);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw cudaException(__FILE__, __LINE__, err);
    }

    cudaUnbindTexture(d_rgba_texture);
}

}
}  // end namespace gpu
