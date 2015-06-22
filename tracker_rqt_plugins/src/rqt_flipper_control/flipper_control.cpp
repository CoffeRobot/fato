/*
 * Copyright (c) 2011, Dirk Thomas, TU Darmstadt
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the TU Darmstadt nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <rqt_flipper_control/flipper_control.h>

#include <pluginlib/class_list_macros.h>
#include <sensor_msgs/JointState.h>

#include <QMessageBox>

namespace rqt_flipper_control {

FlipperControl::FlipperControl()
    : rqt_gui_cpp::Plugin()
    , widget_(0)
{
    setObjectName("FlipperControl");
}

void FlipperControl::initPlugin(qt_gui_cpp::PluginContext& context)
{
    // access standalone command line arguments
    QStringList argv = context.argv();
    // create QWidget
    widget_ = new QWidget();
    // extend the widget with all attributes and children from UI file
    ui_.setupUi(widget_);
    // add widget to the user interface
    context.addWidget(widget_);

    ros::NodeHandle n;

   // flipperFrontSubs = n.subscribe("/flipper/state/front", 1000, &FlipperControl::frontFlipperCallback, this);
   // flipperRearSubs = n.subscribe("/flipper/state/rear", 1000, &FlipperControl::rearFlipperCallback, this);

    connect(ui_.sliderFront,SIGNAL(valueChanged(int)),this,SLOT(sliderFrontChanged(int)));
    connect(ui_.sliderRear,SIGNAL(valueChanged(int)),this,SLOT(sliderRearChanged(int)));

    this->jointStatePub = n.advertise<sensor_msgs::JointState>("/jointstate_cmd", 100);

}

void FlipperControl::shutdownPlugin()
{
    // TODO unregister all publishers here
}

void FlipperControl::saveSettings(qt_gui_cpp::Settings& plugin_settings, qt_gui_cpp::Settings& instance_settings) const
{
    // TODO save intrinsic configuration, usually using:
    // instance_settings.setValue(k, v)
}

void FlipperControl::restoreSettings(const qt_gui_cpp::Settings& plugin_settings, const qt_gui_cpp::Settings& instance_settings)
{

}

void FlipperControl::frontFlipperCallback(const std_msgs::Float64::ConstPtr& msg){
    this->setFlipperFrontPose(msg->data);
}

void FlipperControl::rearFlipperCallback(const std_msgs::Float64::ConstPtr& msg){
    this->setFlipperRearPose(msg->data);
}

void FlipperControl::setFlipperFrontGoal(const int goal){
    ui_.lcdNumFrontGoal->display(goal);
}

void FlipperControl::setFlipperRearGoal(const int goal){
    ui_.lcdNumRearGoal->display(goal);
}


void FlipperControl::setFlipperFrontPose(const int pose){
    ui_.lcdNumFrontPose->display(pose);
}

void FlipperControl::setFlipperRearPose(const int pose){
    ui_.lcdNumRearPose->display(pose);
}


void FlipperControl::sliderFrontChanged(int value){
    float fvalue = value /1000.0;
    sensor_msgs::JointState msg;
    msg.name.push_back("flipper_front");
    msg.position.push_back(fvalue);
    this->jointStatePub.publish(msg);
}

void FlipperControl::sliderRearChanged(int value){

}

void FlipperControl::updateModel(){

}

}

PLUGINLIB_EXPORT_CLASS(rqt_flipper_control::FlipperControl, rqt_gui_cpp::Plugin)
