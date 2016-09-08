

// Std. Includes
#include <string>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GL includes
#include "shader.h"
#include "camera.h"
#include "model.h"
#include "renderer.h"

// GLM Mathemtics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

// Properties
GLuint screenWidth = 640, screenHeight = 480;

// default parameters
int width = 640;
int height = 480;
double fx = 500.0;
double fy = 500.0;
double cx = width / 2.0;
double cy = height / 2.0;
double near_plane = 0.01;   // for init only
double far_plane = 1000.0;  // for init only
float deg2rad = M_PI / 180.0;

// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action,
                  int mode);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void Do_Movement();

// Camera
rendering::Camera camera(glm::vec3(0.0f, 0.0f, 0.0f));
bool keys[1024];
GLfloat lastX = 400, lastY = 300;
bool firstMouse = true;

GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;

// The MAIN function, from here we start our application and run our Game loop
int main() {
  rendering::Renderer renderer(width, height, fx, fy, cx, cy, near_plane,
                               far_plane);
  renderer.initRenderContext(640, 480);

  //  // Init GLFW
  //  glfwInit();
  //  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  //  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  //  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  //  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  //  GLFWwindow* window = glfwCreateWindow(
  //      screenWidth, screenHeight, "LearnOpenGL", nullptr, nullptr);  //
  //      Windowed
  //  glfwMakeContextCurrent(window);

  //  // Set the required callback functions
  //  glfwSetKeyCallback(window, key_callback);
  //  // glfwSetCursorPosCallback(window, mouse_callback);
  //  // glfwSetScrollCallback(window, scroll_callback);

  //  // Options
  //  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  //  // Initialize GLEW to setup the OpenGL Function pointers
  //  glewExperimental = GL_TRUE;
  //  glewInit();

  //  // Define the viewport dimensions
  //  glViewport(0, 0, screenWidth, screenHeight);

  //  // Setup some OpenGL options
  //  glEnable(GL_DEPTH_TEST);

  rendering::Model ourModel(
      "/home/alessandro/projects/rendering_engine/data/models/ros_fuerte/"
      "ros_fuerte.obj");
  rendering::Shader shader(
      "/home/alessandro/projects/rendering_engine/data/shaders/model.vs",
      "/home/alessandro/projects/rendering_engine/data/shaders/model.frag");

  renderer.addModel(
      "/home/alessandro/projects/rendering_engine/data/models/ros_fuerte/"
      "ros_fuerte.obj",
      "/home/alessandro/projects/rendering_engine/data/shaders/model.vs",
      "/home/alessandro/projects/rendering_engine/data/shaders/model.frag");

  glm::mat4 projection = glm::perspective(
      camera.Zoom, (float)screenWidth / (float)screenHeight, 0.1f, 100.0f);

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      cout << projection[i][j] << " ";
    }
    cout << "\n";
  }

  Eigen::Transform<double, 3, Eigen::Affine> pose;
  pose(2, 3) = -1.0f;

  rendering::RigidObject& obj = renderer.getObject(0);
  obj.setVisible(true);
  obj.updatePose(pose);

  // Draw in wireframe
  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  //  glm::mat4 model;
  //  model = glm::translate(
  //      model,
  //      glm::vec3(0.0f, 0.0f, -1.0f));

  //  cout << "model matrix " << endl;
  //  for (int i = 0; i < 4; ++i) {
  //    for (int j = 0; j < 4; ++j) {
  //      cout << model[i][j] << " ";
  //    }
  //    cout << "\n";
  //  }
  //  cout << "object matrix" << endl;
  //  for (int i = 0; i < 4; ++i) {
  //    for (int j = 0; j < 4; ++j) {
  //      cout << obj.model_matrix_[i][j] << " ";
  //    }
  //    cout << "\n";
  //  }

  // Game loop
  while (!glfwWindowShouldClose(renderer.window_)) {
    // Set frame time
    GLfloat currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;



    renderer.render();

    //    shader.use();  // <-- Don't forget this one!
    //    // Transformation matrices
    //    glm::mat4 projection = glm::perspective(
    //        camera.Zoom, (float)screenWidth / (float)screenHeight, 0.1f,
    //        100.0f);
    //    glm::mat4 view = renderer.camera_.GetViewMatrix();
    //    glUniformMatrix4fv(glGetUniformLocation(shader.program_id_,
    //    "projection"),
    //                       1, GL_FALSE,
    //                       glm::value_ptr(renderer.projection_matrix_));
    //    glUniformMatrix4fv(glGetUniformLocation(shader.program_id_, "view"),
    //    1,
    //                       GL_FALSE, glm::value_ptr(view));

    //    // Draw the loaded model
    //    // Translate it down a bit so it's
    //                                        // at the center of the scene

    //    // model = glm::rotate(model, -55.0f, glm::vec3(1.0f, 0.0f, 0.0f));

    //    /* model = glm::scale(
    //        model,
    //        glm::vec3(1.0f, 1.0f,
    //                  1.0f));*/  // It's a bit too big for our scene, so scale
    //                  it
    //                            // down
    //    glUniformMatrix4fv(glGetUniformLocation(shader.program_id_, "model"),
    //    1,
    //                       GL_FALSE, glm::value_ptr(model));
    //    ourModel.draw(shader);

    // Swap the buffers

  }

  glfwTerminate();
  return 0;
}

#pragma region "User input"

// Moves/alters the camera positions based on user input
void Do_Movement() {
  // Camera controls
  if (keys[GLFW_KEY_W]) camera.ProcessKeyboard(rendering::FORWARD, deltaTime);
  if (keys[GLFW_KEY_S]) camera.ProcessKeyboard(rendering::BACKWARD, deltaTime);
  if (keys[GLFW_KEY_A]) camera.ProcessKeyboard(rendering::LEFT, deltaTime);
  if (keys[GLFW_KEY_D]) camera.ProcessKeyboard(rendering::RIGHT, deltaTime);
}

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action,
                  int mode) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);

  if (action == GLFW_PRESS)
    keys[key] = true;
  else if (action == GLFW_RELEASE)
    keys[key] = false;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
  if (firstMouse) {
    lastX = xpos;
    lastY = ypos;
    firstMouse = false;
  }

  GLfloat xoffset = xpos - lastX;
  GLfloat yoffset = lastY - ypos;

  lastX = xpos;
  lastY = ypos;

  camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
  camera.ProcessMouseScroll(yoffset);
}

#pragma endregion
