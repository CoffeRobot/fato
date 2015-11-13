#include "../include/logger.h"

using namespace std;

namespace pinot_tracker{

Logger::Logger()
{
  string homedir = getenv("HOME");
  log_file.open(homedir+"/pinot_tracker_log.txt", std::ofstream::out);
}

void Logger::info(string message)
{
  mutex_.lock();
  log_file << "INFO: " << message << "\n";
  mutex_.unlock();
}

void Logger::warn(string message)
{
  mutex_.lock();
  log_file << "WARN: " << message << "\n";
  mutex_.unlock();
}

void Logger::error(string message)
{
  mutex_.lock();
  log_file << "ERROR: " << message << "\n";
  mutex_.unlock();
}


}

