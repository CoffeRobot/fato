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

#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <string>
#include <ostream>
#include <opencv/cv.h>

#define VERBOSE (0)

class Config
{
public:
	Config() { SetDefaults(); }
	Config(const std::string& path);
	
  // GENERAL CONFIGURATION
  int matching_type;
  bool draw_video;

  // COMMON PARAMETERS
  int num_features;
  int num_octaves;

  // ORB CONFIGURATION
  int orb_num_feature;
  float orb_scale_factor;
  int orb_levels;
  int orb_edge_threshold;
  int orb_first_patch_level;
  int orb_patch_size;

  // BRISK CONFIGURATION
  int brisk_thresh;
  int brisk_octaves;
  float brisk_pattern;

  // AKAZE CONFIGURATION
  int akaze_type;
  int akaze_descriptor_channels;
  float akaze_threshold;
  int akaze_octaves;
  int akaze_sublevels;
  int akaze_descriptor_size;

  // SIFT CONFIGURATION
  int sift_num_features;
  int sift_num_octaves;
  float sift_contrast_threshold;
  int sift_edge_threshold;
  float sift_blur_sigma;

  // CUDA SIFT CONFIGURATION
  int cuda_num_features;
  int cuda_num_octaves;
  float cuda_contrast_threshold;
  float cuda_blur_sigma;
  float cuda_subsampling;

  // FREAK CONFIGURATION
  bool freak_orientation_normalized;
  bool freak_scale_normalized;
  float freak_patter_scale;
  int freak_octaves;

  // SURF CONFIGURATION
  double surf_hessian_threshold;
  int surf_octaves;
  int surf_octave_layers;
  bool surf_extended;
  bool surf_upright;




	friend std::ostream& operator<< (std::ostream& out, const Config& conf);
	
private:
	void SetDefaults();
};

#endif
