/*
Example of a simple camera spawning in flightmare and receiving 100 images

Steps:
1) export RPGQ_PARAM_DIR=~/catkin_plan/src/rpg_flightmare (path to flightmare)
2) launch flightmare build (find latest release here:
https://github.com/uzh-rpg/rpg_flightmare/releases) 3a) if not done yet source
~/catkin_plan/devel/setup.bash 3b) roslaunch example_vision
camera_control.launch 4) images are saved in
RPGQ_PARAM_DIR/examples/example_vision/src/saved_image/
*/

// rpgq simulator
#include <rpgq_simulator/implementation/objects/quadrotor_vehicle/quad_and_rgb_camera.h>
#include <rpgq_simulator/visualization/flightmare_bridge.hpp>
#include <rpgq_simulator/visualization/flightmare_message_types.hpp>

// Eigen
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace RPGQ;

int main(int argc, char* argv[]) {
  // initialize ROS
  ros::init(argc, argv, "camera_control");

  // quad ID can be any real number between
  // 0 ~ 25, each ID corresponding to a unique name
  QuadrotorID quad_ID = 1;
  std::string quad_name = QuadrotorName(quad_ID);
  // create quadrotor with a RGB camera attached.
  std::shared_ptr<Simulator::QuadRGBCamera> quad_rgb =
      std::make_shared<Simulator::QuadRGBCamera>(quad_name, nullptr, 1000000);

  // configure the camera
  std::shared_ptr<Simulator::RGBCamera> rgb_camera = quad_rgb->GetRGBCamera();
  rgb_camera->SetWidth(320);
  rgb_camera->SetHeight(240);

  // set the relative position of the camera with respect to quadrotor center
  // mass
  Eigen::Vector3d B_r_BC(0.0, 0.5, 0.0);
  // set the relative rotation of the camera
  Eigen::Matrix3d R_BC;
  R_BC << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  rgb_camera->SetRelPose(B_r_BC, R_BC);

  // configure the quadrotor
  std::shared_ptr<Simulator::QuadrotorVehicle> quad = quad_rgb->GetQuad();
  Eigen::Vector3d quad_position{0.0, 0.0, 2.0};
  quad->SetPos(quad_position);
  quad->SetQuat(Eigen::Quaterniond(std::cos(0.5 * M_PI_2), 0.0, 0.0,
                                   std::sin(0.5 * M_PI_2)));
  quad->SetSize(Eigen::Vector3d(1, 1, 1));

  // flightmare
  FlightmareTypes::SceneID scene_id = FlightmareTypes::SCENE_WAREHOUSE;
  bool flightmare_ready{false};
  Simulator::RenderMessage_t unity_output;

  // create flightmare birdge and connect sockets..
  std::shared_ptr<Simulator::FlightmareBridge> flightmareBridge_ptr;
  flightmareBridge_ptr = Simulator::FlightmareBridge::getInstance();

  //
  flightmareBridge_ptr->initializeConnections();
  flightmareBridge_ptr->addQuadRGB(quad_rgb);

  // connect to unity.
  // please open the Unity3D standalone.
  double time_out_count = 0;
  double sleep_useconds = 0.2 * 1e5;
  const double connection_time_out = 10.0;  // seconds
  while (!flightmare_ready) {
    if (flightmareBridge_ptr != nullptr) {
      // connect unity
      flightmareBridge_ptr->setScene(scene_id);
      flightmare_ready = flightmareBridge_ptr->connectUnity();
    }
    if (time_out_count / 1e6 > connection_time_out) {
      std::cout << "Flightmare connection failed, time out." << std::endl;
      break;
    }
    // sleep
    usleep(sleep_useconds);
    // increase time out counter
    time_out_count += sleep_useconds;
  }

  // wait 1 seconds. until to environment is fully loaded.
  usleep(1 * 1e6);

  FlightmareTypes::ImgID send_id = 0;
  FlightmareTypes::ImgID receive_id = 0;
  int num_msg = 100;

  cv::Mat rgb_img;

  // main loop
  Timer loopTimer;
  double total_time = 0;
  double min_time = 999;
  double max_time = 0;

  while (true) {
    loopTimer.Reset();

    // Ideally, send_id (pos) == receive_id (image);
    if (receive_id == send_id) {
      send_id += 1;
      //
      // change the quadrotor position and rotation.
      // the camera pos will be changed implicitly
      quad_position << 0, 0, float(send_id) * 0.2;
      quad->SetPos(quad_position);
      quad->SetQuat(Eigen::Quaterniond(std::cos(0.5 * M_PI_2), 0.0, 0.0,
                                       std::sin(0.5 * M_PI_2)));

      // send message to unity (e.g., update quadrotor pose)
      // cannot request image / send camera pose too fast.
      // it takes some time to render images...
      std::cout << "send pose ID: " << send_id << std::endl;
      flightmareBridge_ptr->getRender(send_id);
    }

    // receive message update from Unity3D (e.g. receive image)
    receive_id = flightmareBridge_ptr->handleOutput(unity_output);
    std::cout << "receive image ID: " << receive_id << std::endl;

    //
    rgb_camera->GetRGBImage(rgb_img);

    //
    std::string file_path =
        std::string(getenv("RPGQ_PARAM_DIR")) +
        std::string("/examples/example_vision/src/saved_image/");
    std::string img_string = std::to_string(receive_id) + ".png";
    std::string file_name = file_path + img_string;
    cv::imwrite(file_name, rgb_img);

    double time_elapsed = loopTimer.ElapsedSeconds();
    total_time += time_elapsed;
    min_time = std::min(min_time, time_elapsed);
    max_time = std::max(max_time, time_elapsed);
    if (send_id >= num_msg) break;
  }

  double average_time = total_time / num_msg;
  std::cout << "---\nTotal time: " << total_time
            << "\nAverage: " << average_time << "\nFPS: " << 1 / average_time
            << "\nminimum time: " << min_time << "\nmaximum time: " << max_time
            << std::endl;
  return 0;
}
