/*****************************************************************************/
/*  Copyright (c) 2015, Alessandro Pieropan                                  */
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

#include "VideoReader_x264.h"

VideoReader_x264::VideoReader_x264(const char *filename, int &width, int &height, int &number_of_frames) {

  pFormatCtx = NULL;

  // Register all formats and codecs
  av_register_all();

  // Open video file
  if(avformat_open_input(&pFormatCtx, filename, NULL, NULL)!=0)
    exit(EXIT_FAILURE); // Couldn't open file

  // Retrieve stream information
  if(avformat_find_stream_info(pFormatCtx, NULL)<0)
    exit(EXIT_FAILURE); // Couldn't find stream information

  // Find the first video stream
  videoStream=-1;
  for(int i=0; i<pFormatCtx->nb_streams; i++)
    if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO) {
      videoStream=i;
      break;
    }
  if(videoStream==-1)
    exit(EXIT_FAILURE); // Didn't find a video stream

  AVStream *pVideoStream = pFormatCtx->streams[videoStream];

  // Get a pointer to the codec context for the video stream
  pCodecCtx = pVideoStream->codec;

  // Extract video properties
  number_of_frames = pVideoStream->nb_frames;
  width = pCodecCtx->width;
  height = pCodecCtx->height;

  // Find the decoder for the video stream
  AVCodec *pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
  if(pCodec==NULL) {
    fprintf(stderr, "Unsupported codec!\n");
    exit(EXIT_FAILURE); // Codec not found
  }

  // Open codec
  AVDictionary *optionsDict = NULL;
  if(avcodec_open2(pCodecCtx, pCodec, &optionsDict)<0)
    exit(EXIT_FAILURE); // Could not open codec

  // Allocate video frame
  pFrame = avcodec_alloc_frame();

  // Set-up software scaler (PIX_FMT_YUV420P -> PIX_FMT_BGR24)
  sws_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height, PIX_FMT_BGR24, SWS_BILINEAR, NULL, NULL, NULL);

}


VideoReader_x264::~VideoReader_x264() {

  // Free the YUV frame
  av_free(pFrame);

  // Close the codec
  avcodec_close(pCodecCtx);

  // Close the video file
  avformat_close_input(&pFormatCtx);

}


bool VideoReader_x264::getFrame(uint8_t *buffer) {

  AVPacket packet;

  int frameFinished;
  bool flushingDecoder;

  do {

    flushingDecoder = (av_read_frame(pFormatCtx, &packet)<0);

    if (flushingDecoder) {
      // Create empty packet to flush the decoder
      packet.size = 0;
      packet.data = NULL;
    }

    // Is this a packet from the video stream?
    if(packet.stream_index == videoStream) {

      // Decode video frame
      avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);

      // Did we get a video frame?
      if(frameFinished) {

        // Convert the image from its native format to RGB
        int dst_stride = pCodecCtx->width*3;
        sws_scale(sws_ctx,(uint8_t const * const *)pFrame->data,pFrame->linesize,0,pCodecCtx->height,&buffer,&dst_stride);

      }

    }

    // Free the packet that was allocated by av_read_frame
    av_free_packet(&packet);


  } while ( !(frameFinished || flushingDecoder) );


  return((bool)frameFinished);

}
