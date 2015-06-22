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

#include <rqt_image_cropping/image_cropper.h>

#include <pluginlib/class_list_macros.h>
#include <ros/master.h>
#include <ros/console.h>
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <QMessageBox>
#include <QPainter>

#include <algorithm>

namespace rqt_image_cropping {

ImageCropper::ImageCropper()
    : rqt_gui_cpp::Plugin(), widget_(0), selected_(false) {
  setObjectName("ImageCropper");
}

void ImageCropper::initPlugin(qt_gui_cpp::PluginContext& context) {
  widget_ = new QWidget();
  ui_.setupUi(widget_);

  if (context.serialNumber() > 1) {
    widget_->setWindowTitle(widget_->windowTitle() + " (" +
                            QString::number(context.serialNumber()) + ")");
  }
  context.addWidget(widget_);

  ui_.image_frame->installEventFilter(this);

  updateTopicList();

  ui_.topics_combo_box->setCurrentIndex(ui_.topics_combo_box->findText(""));
  connect(ui_.topics_combo_box, SIGNAL(currentIndexChanged(int)), this,
          SLOT(onInTopicChanged(int)));

  ui_.out_topic_line_edit->setText("/cropped");
  connect(ui_.out_topic_line_edit, SIGNAL(editingFinished()), this,
          SLOT(onOutTopicChanged()));
  onOutTopicChanged();

  ui_.refresh_topics_push_button->setIcon(QIcon::fromTheme("view-refresh"));
  connect(ui_.refresh_topics_push_button, SIGNAL(pressed()), this,
          SLOT(updateTopicList()));

  connect(ui_.dynamic_range_check_box, SIGNAL(toggled(bool)), this,
          SLOT(onDynamicRange(bool)));

  connect(ui_.image_frame, SIGNAL(rightMouseButtonClicked()), this,
          SLOT(onRemoveSelection()));
  connect(ui_.image_frame, SIGNAL(selectionInProgress(QPoint, QPoint)), this,
          SLOT(onSelectionInProgress(QPoint, QPoint)));
  connect(ui_.image_frame, SIGNAL(selectionFinished(QPoint, QPoint)), this,
          SLOT(onSelectionFinished(QPoint, QPoint)));

  nh_ = getNodeHandle();

  tracker_gui_publisher_ = nh_.advertise<std_msgs::String>("pinot_tracker_bbox",1);
}

bool ImageCropper::eventFilter(QObject* watched, QEvent* event) {
  if (watched == ui_.image_frame && event->type() == QEvent::Paint) {
    QPainter painter(ui_.image_frame);
    if (!qimage_.isNull()) {
      ui_.image_frame->resizeToFitAspectRatio();
      // TODO: check if full draw is really necessary
      // QPaintEvent* paint_event = dynamic_cast<QPaintEvent*>(event);
      // painter.drawImage(paint_event->rect(), qimage_);
      qimage_mutex_.lock();
      painter.drawImage(ui_.image_frame->contentsRect(), qimage_);
      qimage_mutex_.unlock();
    } else {
      // default image with gradient
      QLinearGradient gradient(0, 0, ui_.image_frame->frameRect().width(),
                               ui_.image_frame->frameRect().height());
      gradient.setColorAt(0, Qt::white);
      gradient.setColorAt(1, Qt::black);
      painter.setBrush(gradient);
      painter.drawRect(0, 0, ui_.image_frame->frameRect().width() + 1,
                       ui_.image_frame->frameRect().height() + 1);
    }

    if (selected_) {
      selection_ = QRectF(selection_top_left_rect_, selection_size_rect_);

      painter.setPen(Qt::red);
      painter.drawRect(selection_);
    }

    ui_.image_frame->update();

    return false;
  }

  return QObject::eventFilter(watched, event);
}

void ImageCropper::shutdownPlugin() {
  subscriber_.shutdown();
  publisher_.shutdown();
}

void ImageCropper::saveSettings(qt_gui_cpp::Settings& plugin_settings,
                                qt_gui_cpp::Settings& instance_settings) const {
  QString topic = ui_.topics_combo_box->currentText();
  // qDebug("ImageCropper::saveSettings() topic '%s'",
  // topic.toStdString().c_str());
  instance_settings.setValue("topic", topic);
  instance_settings.setValue("dynamic_range",
                             ui_.dynamic_range_check_box->isChecked());
  instance_settings.setValue("max_range",
                             ui_.max_range_double_spin_box->value());
}

void ImageCropper::restoreSettings(
    const qt_gui_cpp::Settings& plugin_settings,
    const qt_gui_cpp::Settings& instance_settings) {
  bool dynamic_range_checked =
      instance_settings.value("dynamic_range", false).toBool();
  ui_.dynamic_range_check_box->setChecked(dynamic_range_checked);

  double max_range =
      instance_settings.value("max_range",
                              ui_.max_range_double_spin_box->value())
          .toDouble();
  ui_.max_range_double_spin_box->setValue(max_range);

  QString topic = instance_settings.value("topic", "").toString();
  // qDebug("ImageCropper::restoreSettings() topic '%s'",
  // topic.toStdString().c_str());
  selectTopic(topic);
}

void ImageCropper::updateTopicList() {
  QSet<QString> message_types;
  message_types.insert("sensor_msgs/Image");

  // get declared transports
  QList<QString> transports;
  image_transport::ImageTransport it(getNodeHandle());
  std::vector<std::string> declared = it.getDeclaredTransports();
  for (std::vector<std::string>::const_iterator it = declared.begin();
       it != declared.end(); it++) {
    // qDebug("ImageCropper::updateTopicList() declared transport '%s'",
    // it->c_str());
    QString transport = it->c_str();

    // strip prefix from transport name
    QString prefix = "image_transport/";
    if (transport.startsWith(prefix)) {
      transport = transport.mid(prefix.length());
    }
    transports.append(transport);
  }

  QString selected = ui_.topics_combo_box->currentText();

  // fill combo box
  QList<QString> topics = getTopicList(message_types, transports);
  topics.append("");
  qSort(topics);
  ui_.topics_combo_box->clear();
  for (QList<QString>::const_iterator it = topics.begin(); it != topics.end();
       it++) {
    QString label(*it);
    label.replace(" ", "/");
    ui_.topics_combo_box->addItem(label, QVariant(*it));
  }

  // restore previous selection
  selectTopic(selected);
}

QList<QString> ImageCropper::getTopicList(const QSet<QString>& message_types,
                                          const QList<QString>& transports) {
  ros::master::V_TopicInfo topic_info;
  ros::master::getTopics(topic_info);

  QSet<QString> all_topics;
  for (ros::master::V_TopicInfo::const_iterator it = topic_info.begin();
       it != topic_info.end(); it++) {
    all_topics.insert(it->name.c_str());
  }

  QList<QString> topics;
  for (ros::master::V_TopicInfo::const_iterator it = topic_info.begin();
       it != topic_info.end(); it++) {
    if (message_types.contains(it->datatype.c_str())) {
      QString topic = it->name.c_str();

      // add raw topic
      topics.append(topic);
      // qDebug("ImageCropper::getTopicList() raw topic '%s'",
      // topic.toStdString().c_str());

      // add transport specific sub-topics
      for (QList<QString>::const_iterator jt = transports.begin();
           jt != transports.end(); jt++) {
        if (all_topics.contains(topic + "/" + *jt)) {
          QString sub = topic + " " + *jt;
          topics.append(sub);
          // qDebug("ImageCropper::getTopicList() transport specific sub-topic
          // '%s'", sub.toStdString().c_str());
        }
      }
    }
  }
  return topics;
}

void ImageCropper::selectTopic(const QString& topic) {
  int index = ui_.topics_combo_box->findText(topic);
  if (index == -1) {
    index = ui_.topics_combo_box->findText("");
  }
  ui_.topics_combo_box->setCurrentIndex(index);
}

void ImageCropper::onInTopicChanged(int index) {
  subscriber_.shutdown();

  // reset image on topic change
  qimage_ = QImage();
  ui_.image_frame->update();

  QStringList parts =
      ui_.topics_combo_box->itemData(index).toString().split(" ");
  QString topic = parts.first();
  QString transport = parts.length() == 2 ? parts.last() : "raw";

  if (!topic.isEmpty()) {
    image_transport::ImageTransport it(getNodeHandle());
    image_transport::TransportHints hints(transport.toStdString());
    try {
      subscriber_ = it.subscribeCamera(
          topic.toStdString(), 1, &ImageCropper::callbackImage, this, hints);
      // qDebug("ImageCropper::onInTopicChanged() to topic '%s' with transport
      // '%s'", topic.toStdString().c_str(),
      // subscriber_.getTransport().c_str());
    }
    catch (image_transport::TransportLoadException& e) {
      QMessageBox::warning(widget_, tr("Loading image transport plugin failed"),
                           e.what());
    }
  }

  selected_ = false;
}

void ImageCropper::onOutTopicChanged() {
  publisher_.shutdown();

  QString topic = ui_.out_topic_line_edit->text();

  if (!topic.isEmpty()) {
    image_transport::ImageTransport it(getNodeHandle());

    try {
      publisher_ = it.advertiseCamera(topic.toStdString(), 1, true);
    }
    catch (image_transport::TransportLoadException& e) {
      QMessageBox::warning(widget_, tr("Loading image transport plugin failed"),
                           e.what());
    }
  }

  selected_ = false;
}

void ImageCropper::onSelectionInProgress(QPoint p1, QPoint p2) {
  enforceSelectionConstraints(p1);
  enforceSelectionConstraints(p2);

  int tl_x = std::min(p1.x(), p2.x());
  int tl_y = std::min(p1.y(), p2.y());

  selection_top_left_rect_ = QPointF(tl_x, tl_y);
  selection_size_rect_ = QSizeF(abs(p1.x() - p2.x()), abs(p1.y() - p2.y()));

  // ROS_DEBUG_STREAM << "p1: " << p1.x() << " " << p1.y() << " p2: " << p2.x()
  // << " " << p2.y();

  selected_ = true;
}

void ImageCropper::onSelectionFinished(QPoint p1, QPoint p2) {
  enforceSelectionConstraints(p1);
  enforceSelectionConstraints(p2);

  int tl_x = p1.x() < p2.x() ? p1.x() : p2.x();
  int tl_y = p1.y() < p2.y() ? p1.y() : p2.y();

  selection_top_left_rect_ = QPointF(tl_x, tl_y);
  selection_size_rect_ = QSizeF(abs(p1.x() - p2.x()), abs(p1.y() - p2.y()));

  selection_top_left_ = QPointF(tl_x, tl_y);
  selection_size_ = QSizeF(abs(p1.x() - p2.x()), abs(p1.y() - p2.y()));

  selection_top_left_ *=
      (double)qimage_.width() /
      (double)ui_.image_frame->contentsRect().width();  // width();
  selection_size_ *= (double)qimage_.width() / (double)ui_.image_frame->width();

  // crop image from cv image
  cv::Mat roi =
      cv::Mat(conversion_mat_,
              cv::Rect(selection_top_left_.x(), selection_top_left_.y(),
                       selection_size_.width(), selection_size_.height()));

  //    std::cout << "sle height: " << selection_size_.height() << " width: " <<
  // selection_size_.width() << std::endl;
  //    std::cout << "roi rows: " << roi.rows << " cols: " << roi.cols <<
  // std::endl;

  //    cv_bridge::CvImage crop;
  //    crop.header = sens_msg_image_->header;
  //    crop.encoding = sensor_msgs::image_encodings::RGB8;
  //    crop.image = roi;

  // adapt camera info
  // Create updated CameraInfo message
  sensor_msgs::CameraInfoPtr out_info =
      boost::make_shared<sensor_msgs::CameraInfo>(*camera_info_);
  int binning_x = std::max((int)camera_info_->binning_x, 1);
  int binning_y = std::max((int)camera_info_->binning_y, 1);
  out_info->binning_x = binning_x * 1;
  out_info->binning_y = binning_y * 1;
  out_info->roi.x_offset += selection_top_left_.x() * binning_x;
  out_info->roi.y_offset += selection_top_left_.y() * binning_y;
  out_info->roi.height = selection_size_.height() * binning_y;
  out_info->roi.width = selection_size_.width() * binning_x;

  if (publisher_.getNumSubscribers()) {
    publisher_.publish(sens_msg_image_, out_info);
  }
}

void ImageCropper::onRemoveSelection() {
  selected_region_ = QImage();
  selected_ = false;
}

void ImageCropper::enforceSelectionConstraints(QPoint& p) {
  int min_x = 1;
  int max_x = ui_.image_frame->width() - 2 * ui_.image_frame->frameWidth();

  int min_y = 1;
  int max_y = ui_.image_frame->height() - 2 * ui_.image_frame->frameWidth();

  p.setX(std::min(std::max(p.x(), min_x), max_x));
  p.setY(std::min(std::max(p.y(), min_y), max_y));
}

void ImageCropper::onDynamicRange(bool checked) {
  ui_.max_range_double_spin_box->setEnabled(!checked);
}

void ImageCropper::callbackImage(const sensor_msgs::Image::ConstPtr& img,
                                 const sensor_msgs::CameraInfoConstPtr& ci) {

  sens_msg_image_ = img;
  camera_info_ = ci;

  try {
    // First let cv_bridge do its magic
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvShare(img, sensor_msgs::image_encodings::RGB8);
    conversion_mat_ = cv_ptr->image;
  }
  catch (cv_bridge::Exception& e) {
    // If we're here, there is no conversion that makes sense, but let's try to
    // imagine a few first
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(img);
    if (img->encoding == "CV_8UC3") {
      // assuming it is rgb
      conversion_mat_ = cv_ptr->image;
    } else if (img->encoding == "8UC1") {
      // convert gray to rgb
      cv::cvtColor(cv_ptr->image, conversion_mat_, CV_GRAY2RGB);
    } else if (img->encoding == "16UC1" || img->encoding == "32FC1") {
      // scale / quantify
      image_min_value_ = 0;
      image_max_value_ = ui_.max_range_double_spin_box->value();
      if (img->encoding == "16UC1") image_max_value_ *= 1000;
      if (ui_.dynamic_range_check_box->isChecked()) {
        // dynamically adjust range based on min/max in image
        cv::minMaxLoc(cv_ptr->image, &image_min_value_, &image_max_value_);
        if (image_min_value_ == image_max_value_) {
          // completely homogeneous images are displayed in gray
          image_min_value_ = 0;
          image_max_value_ = 2;
        }
      }
      cv::Mat img_scaled_8u;
      cv::Mat(cv_ptr->image - image_min_value_).convertTo(
          img_scaled_8u, CV_8UC1, 255. / (image_max_value_ - image_min_value_));
      cv::cvtColor(img_scaled_8u, conversion_mat_, CV_GRAY2RGB);
    } else {
      qWarning(
          "ImageCropper.callback_image() could not convert image from '%s' to "
          "'rgb8' (%s)",
          img->encoding.c_str(), e.what());
      qimage_ = QImage();
      return;
    }
  }

  // copy temporary image as it uses the conversion_mat_ for storage which is
  // asynchronously overwritten in the next callback invocation
  QImage image(conversion_mat_.data, conversion_mat_.cols, conversion_mat_.rows,
               QImage::Format_RGB888);
  qimage_mutex_.lock();
  qimage_ = image.copy();
  qimage_mutex_.unlock();

  ui_.image_frame->setAspectRatio(qimage_.width(), qimage_.height());
  // onZoom1(false);

  ui_.image_frame->setInnerFrameMinimumSize(QSize(80, 60));
  ui_.image_frame->setMaximumSize(QSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX));
  widget_->setMinimumSize(QSize(80, 60));
  widget_->setMaximumSize(QSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX));
}

void ImageCropper::publishCrop() {

  ROS_INFO("Publishing crop information");
}
}

PLUGINLIB_EXPORT_CLASS(rqt_image_cropping::ImageCropper, rqt_gui_cpp::Plugin)
