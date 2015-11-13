#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <mutex>
#include <string>
#include <sstream>
#include <memory>

namespace pinot_tracker{

class Logger
{
public:
  static std::unique_ptr<Logger>& getInstance()
  {
    static std::unique_ptr<Logger> instance(new Logger());

    return instance;
  }

  void info(std::string message);
  void warn(std::string message);
  void error(std::string message);


private:
  Logger();

  std::ofstream log_file;
  std::mutex mutex_;
};

}

#endif // LOGGER_H
