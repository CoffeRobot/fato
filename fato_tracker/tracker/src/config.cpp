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

#include "../include/config.h"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

Config::Config(const std::string& path)
{
	SetDefaults();
	
	ifstream f(path.c_str());
	if (!f)
	{
		cout << "error: could not load config file: " << path << endl;
		return;
	}
	
	string line, name, tmp;
	while (getline(f, line))
	{
		istringstream iss(line);
		iss >> name >> tmp;

		// skip invalid lines and comments
		if (iss.fail() || tmp != "=" || name[0] == '#') continue;

    if (name == "orb_num_features") iss >> orb_num_feature;
    else if (name == "orb_scale_factor") iss >> orb_scale_factor;
    else if (name == "orb_levels") iss >> orb_levels;
    else if (name == "orb_edge_threshold") iss >> orb_edge_threshold;
    else if (name == "orb_first_level") iss >> orb_first_patch_level;
    else if (name == "orb_patch_size") iss >> orb_patch_size;
    else if (name == "brisk_tresh") iss >> brisk_thresh;
    else if (name == "brisk_octaves") iss >> brisk_octaves;
    else if (name == "brisk_patter") iss >> brisk_pattern;
    else if (name == "akaze_type") iss >> akaze_type;
    else if (name == "akaze_descriptor_channels") iss >> akaze_descriptor_channels;
    else if (name == "akaze_threshold") iss >> akaze_threshold;
    else if (name == "akaze_octaves") iss >> akaze_octaves;
    else if (name == "akaze_sublevels") iss >> akaze_sublevels;
    else if (name == "akaze_descriptor_size") iss >> akaze_descriptor_size;
    else if (name == "sift_num_features") iss >> sift_num_features;
    else if (name == "sift_num_octaves") iss >> sift_num_octaves;
    else if (name == "sift_contrast_threshold") iss >> sift_contrast_threshold;
    else if (name == "sift_edge_threshold") iss >> sift_edge_threshold;
    else if (name == "sift_blur_sigma") iss >> sift_blur_sigma;
    else if (name == "cuda_num_features") iss >> cuda_num_features;
    else if (name == "cuda_num_octaves") iss >> cuda_num_octaves;
    else if (name == "cuda_contrast_threshold") iss >> cuda_contrast_threshold;
    else if (name == "cuda_blur_sigma") iss >> cuda_blur_sigma;
    else if (name == "cuda_subsampling") iss >> cuda_subsampling;
    else if (name == "freak_octaves") iss >> freak_octaves;
    else if (name == "freak_orientation_normalized") iss >> freak_orientation_normalized;
    else if (name == "freak_patter_scale") iss >> freak_patter_scale;
    else if (name == "freak_scale_normalized") iss >> freak_scale_normalized;
    else if (name == "surf_extended") iss >> surf_extended;
    else if (name == "surf_hessian_threshold") iss >> surf_hessian_threshold;
    else if (name == "surf_octave_layers") iss >> surf_octave_layers;
    else if (name == "surf_octaves") iss >> surf_octaves;
    else if (name == "surf_upright") iss >> surf_upright;
    else if (name == "matching_type") iss >> matching_type;
    else if (name == "draw_video") iss >> draw_video;
    else if (name == "num_features") iss >> num_features;
    else if (name == "num_octaves") iss >> num_octaves;
	}
}

void Config::SetDefaults()
{

  // GENERAL CONFIGURATION
  matching_type = 0;
  draw_video = false;
  num_features = 2500;
  num_octaves = 4;

  // ORB CONFIGURATION
  orb_num_feature = 500;
  orb_scale_factor = 1.2f;
  orb_levels = 8;
  orb_edge_threshold = 31;
  orb_first_patch_level = 0;
  orb_patch_size = 31;

  // BRISK CONFIGURATION
  brisk_thresh = 30;
  brisk_octaves = 3;
  brisk_pattern = 1.0f;

  // AKAZE CONFIGURATION
  akaze_type = 2;
  akaze_descriptor_channels = 3;
  akaze_threshold = 0.001f;
  akaze_octaves = 4;
  akaze_sublevels = 4;
  akaze_descriptor_size = 256;

  // SIFT CONFIGURATION
  sift_num_features = 2500;
  sift_num_octaves = 3;
  sift_contrast_threshold = 0.04f;
  sift_edge_threshold = 10;
  sift_blur_sigma = 1.0f;

  // CUDA SIFT CONFIGURATION
  cuda_num_features = 2500;
  cuda_num_octaves = 3;
  cuda_contrast_threshold = 2.0f;
  cuda_blur_sigma = 0.0f;
  cuda_subsampling = 0.0f;

}


ostream& operator<< (ostream& out, const Config& conf)
{
  out << "CONFIGURATION PARAMETERS:" << endl;
  out << " orb_edge_threshold      = " << conf.orb_edge_threshold << endl;
  out << " orb_first_patch_level   = " << conf.orb_first_patch_level << endl;
  out << " orb_levels              = " << conf.orb_levels << endl;
  out << " orb_num_feature         = " << conf.orb_num_feature << endl;
  out << " orb_patch_size          = " << conf.orb_patch_size << endl;
  out << " orb_scale_factor        = " << conf.orb_scale_factor << endl;
  out << " brisk_octaves           = " << conf.brisk_octaves << endl;
  out << " brisk_pattern           = " << conf.brisk_pattern << endl;
  out << " brisk_thresh            = " << conf.brisk_thresh << endl;
  out << " sift_blur_sigma         = " << conf.sift_blur_sigma << endl;
  out << " sift_contrast_threshold = " << conf.sift_contrast_threshold << endl;
  out << " sift_edge_threshold     = " << conf.sift_edge_threshold << endl;
  out << " sift_num_features       = " << conf.sift_num_features << endl;
  out << " sift_num_octaves        = " << conf.sift_num_octaves << endl;
  out << " cuda_blur_sigma         = " << conf.cuda_blur_sigma << endl;
  out << " cuda_contrast_threshold = " << conf.cuda_contrast_threshold << endl;
  out << " cuda_num_features       = " << conf.cuda_num_features << endl;
  out << " cuda_num_octaves        = " << conf.cuda_num_octaves << endl;
  out << " cuda_num_subsampling    = " << conf.cuda_subsampling << endl;
  out << " freak_octaves           = " << conf.freak_octaves << endl;
  out << " freak_orientation_norm  = " << conf.freak_orientation_normalized << endl;
  out << " freak_patter_scale      = " << conf.freak_patter_scale << endl;
  out << " freak_scale_normalized  = " << conf.freak_scale_normalized << endl;
  out << " surf_extended           = " << conf.surf_extended << endl;
  out << " surf_hessian_threshold  = " << conf.surf_hessian_threshold << endl;
  out << " surf_octave_layers      = " << conf.surf_octave_layers << endl;
  out << " surf_octaves            = " << conf.surf_octaves << endl;
  out << " surf_upright            = " << conf.surf_upright << endl;
  out << " num_octaves             = " << conf.num_octaves << endl;
  out << " num_features            = " << conf.num_features << endl;
  out << " akaze type              = " << conf.akaze_type << endl;
  out << " akaze channels          = " << conf.akaze_descriptor_channels << endl;
  out << " akaze threshold         = " << conf.akaze_threshold << endl;
  out << " akaze octaves           = " << conf.akaze_octaves << endl;
  out << " akaze sublevels         = " << conf.akaze_sublevels << endl;
  out << " akaze descriptor size   = " << conf.akaze_descriptor_size << endl;
	
	return out;
}
