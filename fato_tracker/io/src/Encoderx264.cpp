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

#include "../include/Encoderx264.h"
#include <stdio.h>
#include <iostream>

using namespace std;

namespace pinot_tracker {

bool Encoderx264::initialized = false;
mutex Encoderx264::initMutex;

Encoderx264::Encoderx264(const char *filename, int width, int height,
                         int frame_rate, string codec_name, int crf,
                         const char *preset) {

  init();

  // find video encoder
  m_pCodec = avcodec_find_encoder_by_name(codec_name.c_str());
  if (!m_pCodec) {
    cerr << "Codec not found!" << endl;
    exit(EXIT_FAILURE);
  }

  /* auto detect the output format from the name */
  AVOutputFormat *fmt = av_guess_format(NULL, filename, NULL);
  if (!fmt) {
    cerr << "Could not find suitable output format" << endl;
    exit(EXIT_FAILURE);
  }

  /* allocate the output media context */
  m_pFormatContext = avformat_alloc_context();
  if (!m_pFormatContext) {
    fprintf(stderr, "Memory error\n");
    exit(EXIT_FAILURE);
  }
  m_pFormatContext->oformat = fmt;
// out->filename = filename;
#ifdef _WIN32
  _snprintf(m_pFormatContext->filename, sizeof(m_pFormatContext->filename),
            "%s", filename);
#else
  snprintf(m_pFormatContext->filename, sizeof(m_pFormatContext->filename), "%s",
           filename);
#endif

  if (avio_open2(&m_pFormatContext->pb, m_pFormatContext->filename,
                 AVIO_FLAG_WRITE, NULL, NULL) < 0) {
    fprintf(stderr, "Could not open '%s'\n", m_pFormatContext->filename);
    exit(EXIT_FAILURE);
  }

  AVStream *st = avformat_new_stream(m_pFormatContext, m_pCodec);
  if (!st) {
    fprintf(stderr, "Could not alloc stream\n");
    exit(EXIT_FAILURE);
  }

  st->sample_aspect_ratio.den = 1;
  st->sample_aspect_ratio.num = 1;

  m_pEncContext = st->codec;

  m_pEncContext->width = width;
  m_pEncContext->height = height;

  m_pEncContext->time_base.den = frame_rate;
  m_pEncContext->time_base.num = 1;
  if (codec_name.compare("ffv1") == 0)
    m_pEncContext->pix_fmt = PIX_FMT_YUV420P;
  else
    m_pEncContext->pix_fmt = PIX_FMT_YUV420P;

  m_pEncContext->sample_aspect_ratio.den = 1;
  m_pEncContext->sample_aspect_ratio.num = 1;

  m_pEncContext->thread_count = 0; /* use several threads for encoding */

  AVDictionary *opts = NULL;
  char crf_str[4];
  sprintf(crf_str, "%d", crf);
  av_dict_set(&opts, "crf", crf_str, 0);
  //  av_dict_set(&opts, "preset", "superfast", 0);
  av_dict_set(&opts, "preset", preset, 0);

  /* open the codec */
  if (avcodec_open2(m_pEncContext, m_pCodec, &opts) < 0) {
    fprintf(stderr, "could not open codec\n");
    exit(EXIT_FAILURE);
  }

  /* allocate output buffer */
  video_outbuf_size =
      m_pEncContext->width * m_pEncContext->height * 4; /* upper bound */
  video_outbuf = (int *)av_malloc(video_outbuf_size);
  if (!video_outbuf) {
    fprintf(stderr, "Alloc outbuf fail\n");
    exit(EXIT_FAILURE);
  }
  av_dump_format(m_pFormatContext, 0, m_pFormatContext->filename, 1);

  int r;
  r = avformat_write_header(m_pFormatContext, NULL);
  if (r) {
    fprintf(stderr, "write out file fail\n");
    exit(r);
  }

  AVRational ratio = av_div_q(m_pFormatContext->streams[0]->time_base,
                              m_pEncContext->time_base);
  pts_step = ratio.den / ratio.num;

  frames_in = 0;
  frames_out = 0;

  if (codec_name.compare("ffv1") == 0) {
    sws_ctx = sws_getContext(m_pEncContext->width, m_pEncContext->height,
                             PIX_FMT_GRAY16LE, m_pEncContext->width,
                             m_pEncContext->height, m_pEncContext->pix_fmt,
                             SWS_BILINEAR, NULL, NULL, NULL);

  } else {
    sws_ctx = sws_getContext(m_pEncContext->width, m_pEncContext->height,
                             PIX_FMT_BGR24, m_pEncContext->width,
                             m_pEncContext->height, m_pEncContext->pix_fmt,
                             SWS_BILINEAR, NULL, NULL, NULL);
  }

  pFrame = avcodec_alloc_frame();
  int numBytes = avpicture_get_size(
      m_pEncContext->pix_fmt, m_pEncContext->width, m_pEncContext->height);
  buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
  avpicture_fill((AVPicture *)pFrame, buffer, m_pEncContext->pix_fmt,
                 m_pEncContext->width, m_pEncContext->height);

  m_pPktOut = new AVPacket;
  av_init_packet(m_pPktOut);
  m_pPktOut->data = 0;

#ifdef VERBOSE_LOGGING
  cout << "Encoder correctly initialized!" << endl;
#endif

}

Encoderx264::~Encoderx264() {

  av_free(buffer);
  av_free(pFrame);

  while (1) {
    av_free_packet(m_pPktOut);
    int got_packet = 0;
    int ret =
        avcodec_encode_video2(m_pEncContext, m_pPktOut, NULL, &got_packet);

    if (got_packet)
      ret = av_interleaved_write_frame(m_pFormatContext, m_pPktOut);
    else
      break;
  }

  av_write_trailer(m_pFormatContext);
  av_dump_format(m_pFormatContext, 0, m_pFormatContext->filename, 1);

  avcodec_close(m_pEncContext);
  av_free(video_outbuf);
  avio_close(m_pFormatContext->pb);
  avformat_free_context(m_pFormatContext);
}

void Encoderx264::addFrame(uint8_t *buffer) {

  // convert frame to encoder format (PIX_FMT_YUV420P)
  int src_stride;
  if (m_pEncContext->codec_id == AV_CODEC_ID_FFV1) {
    src_stride = m_pEncContext->width;
  } else
    src_stride = m_pEncContext->width * 3;

  sws_scale(sws_ctx, &buffer, &src_stride, 0, m_pEncContext->height,
            pFrame->data, pFrame->linesize);

  pFrame->width = m_pEncContext->width;
  pFrame->height = m_pEncContext->height;
  pFrame->pts = frames_in++;

  // encode the image
  av_free_packet(m_pPktOut);
  int got_packet = 0;
  int ret =
      avcodec_encode_video2(m_pEncContext, m_pPktOut, pFrame, &got_packet);

  if (ret < 0) {
    cerr << "Error encoding a frame\n";
  }

  if (got_packet) {
    // av_packet_rescale_ts(m_pPktOut, m_pEncContext->time_base,
    // m_pEncContext->)
    ret = av_interleaved_write_frame(m_pFormatContext, m_pPktOut);
  }

  if (ret != 0) {
    cerr << "Error while writing video frame\n";
    exit(1);
  }
}

void Encoderx264::init() {
  // needs to be protected since the library could be initialized from multiple
  // threads simultaneously
  Encoderx264::initMutex.lock();
  if (!initialized) {
    printf("initializing lock manager\n");
    av_lockmgr_register(&lockManager);
    av_register_all();
    initialized = true;
  }
  Encoderx264::initMutex.unlock();
}

int Encoderx264::lockManager(void **mutex, enum AVLockOp op) {
  if (NULL == mutex) return -1;

  switch (op) {
    case AV_LOCK_CREATE: {
      *mutex = NULL;
      std::mutex *m = new std::mutex();
      *mutex = static_cast<void *>(m);
      break;
    }
    case AV_LOCK_OBTAIN: {
      std::mutex *m = static_cast<std::mutex *>(*mutex);
      m->lock();
      break;
    }
    case AV_LOCK_RELEASE: {
      std::mutex *m = static_cast<std::mutex *>(*mutex);
      m->unlock();
      break;
    }
    case AV_LOCK_DESTROY: {
      std::mutex *m = static_cast<std::mutex *>(*mutex);
      delete m;
      break;
    }
    default:
      break;
  }
  return 0;
}

}  // end namespace
