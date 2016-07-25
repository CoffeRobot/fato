/*****************************************************************************/
/*  Copyright (c) 2016, Alessandro Pieropan                                  */
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

#include "../../fato_rendering/include/multiple_rigid_models_ogre.h"
#include "../../fato_rendering/include/windowless_gl_context.h"
#include <string>


using namespace std;

int main(int argc, char** argv)
{
    int image_w = 640;
    int image_h = 480;
    double focal_length_x = 632.7361080533549;
    double focal_length_y = 634.2327075892116;
    double nodal_point_x = 321.9474832561696;
    double nodal_point_y = 223.9353111003978;
    string filename = "/home/alessandro/projects/drone_ws/src/fato_tracker/data/ros_hydro/ros_hydro.obj";


    pose::MultipleRigidModelsOgre rendering_engine(
            image_w, image_h, focal_length_x,
            focal_length_y, nodal_point_x, nodal_point_y, 0.01, 10.0);

    rendering_engine.addModel(filename);

    Eigen::Vector3d translation(0, 0, 0.7);
    Eigen::Vector3d rotation(M_PI, 1.3962634015954636, 0);


    double T[] = {translation[0], translation[1], translation[2]};
    double R[] = {translation[0], translation[1], translation[2]};
    std::vector<pose::TranslationRotation3D> TR(1);
    TR.at(0) = pose::TranslationRotation3D(T, R);

    rendering_engine.render(TR);

    model_ogre.getTexture();

}
