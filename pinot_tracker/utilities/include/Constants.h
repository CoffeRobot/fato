#pragma once

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

// static const float FOCAL = 525.0f;
// static const float BASELINE = 83.0f;
