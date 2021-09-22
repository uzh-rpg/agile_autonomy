#include <experimental/filesystem>
#include <fstream>

#include "quadrotor_common/quad_state_estimate.h"
#include "quadrotor_common/trajectory.h"
#include "ros/ros.h"
#include "std_msgs/Int8.h"

#include "agile_autonomy_utils/generate_reference.h"
#include "traj_sampler/traj_sampler.h"

namespace fs = std::experimental::filesystem;

namespace generate_label {

class GenerateLabel {
 public:
  GenerateLabel(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

  GenerateLabel() : GenerateLabel(ros::NodeHandle(), ros::NodeHandle("~")) {}

  virtual ~GenerateLabel();

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber generate_label_sub_;
  ros::Subscriber perform_mpc_optim_sub_;
  ros::Subscriber thread_completed_sub_;
  ros::Publisher mpc_completed_pub_;
  ros::Publisher completed_labelling_pub_;
  ros::Publisher thread_completed_pub_;
  std::vector<std::string> getDirectories(const std::string& s);
  void generateInitialGuesses(const std_msgs::BoolConstPtr& msg);
  void generateLabelCallback(const std_msgs::BoolConstPtr& msg);
  void completedThreadCallback(const std_msgs::Int8ConstPtr& msg);
  bool loadParameters();

  rpg_mpc::MpcController<double> init_controller_ =
      rpg_mpc::MpcController<double>(ros::NodeHandle(), ros::NodeHandle("~"),
                                     "vio_mpc_path");
  rpg_mpc::MpcParams<double> init_controller_params_;

  std::string data_dir_;
  int traj_len_;
  double traj_dt_;
  double rand_theta_;
  double rand_phi_;
  unsigned int continuity_order_;
  int max_steps_metropolis_;
  int save_n_best_;
  double save_max_cost_;
  double crash_dist_;
  double crash_penalty_;
  Eigen::Vector3d drone_dimensions_;
  bool perform_mpc_optimization_;
  bool perform_global_planning_;
  bool save_wf_, save_bf_, save_ga_;

  int start_idx_;
  int end_idx_;

  int bspline_anchors_;

  int n_threads_;
  int thread_idx_;
  int num_completed_threads_ = 0;
  bool verbose_ = false;
};
}  // namespace generate_label
