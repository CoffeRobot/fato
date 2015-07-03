#pragma once
namespace pinot {

namespace gpu {

enum struct Status {
  INIT = 0,
  MATCH,
  NOMATCH,
  TRACK,
  BOTH,
  BACKGROUND,
  LOST,
  NOCLUSTER,
  LEARN
};

enum FACE : int {
  LEFT = 0,
  RIGHT,
  TOP,
  DOWN,
  FRONT,
  BACK
};
}
}
