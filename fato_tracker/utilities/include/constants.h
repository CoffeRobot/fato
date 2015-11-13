#ifndef CONSTANTS_H
#define CONSTANTS_H

//FIXME: fix error related to namespace here

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

#endif

// static const float FOCAL = 525.0f;
// static const float BASELINE = 83.0f;
