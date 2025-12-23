#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <vector>
#include <cmath>
#include <algorithm>

class SonarOccupancyMapper {
public:
    SonarOccupancyMapper() : tf_listener_(tf_buffer_) {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");

        // --- 参数配置 ---
        pnh.param<std::string>("map_frame", map_frame_, "world");
        pnh.param<std::string>("robot_base_frame", robot_base_frame_, "rexrov2/base_link");
        
        pnh.param("map_res", map_res_, 0.2f);
        pnh.param("map_width_m", map_width_m_, 300.0f);
        pnh.param("map_height_m", map_height_m_, 300.0f);
        pnh.param("map_origin_x", map_origin_x_, -150.0f);
        pnh.param("map_origin_y", map_origin_y_, -150.0f);

        pnh.param("sonar_max_range", sonar_max_range_, 16.0f);
        float fov_deg;
        pnh.param("sonar_fov_deg", fov_deg, 95.0f);
        sonar_fov_rad_ = fov_deg * M_PI / 180.0f;

        pnh.param("hit_increment", hit_increment_, 30.0f);
        pnh.param("max_occupancy", max_occupancy_, 100.0f);
        
        pnh.param("base_decay_rate", base_decay_rate_, 3.0f);
        pnh.param("velocity_decay_factor", velocity_decay_factor_, 1.0f);
        
        pnh.param("dilation_radius", dilation_radius_, 1);
        pnh.param("max_tilt_deg", max_tilt_deg_, 5.0f);

        // --- 内部变量初始化 ---
        grid_width_ = static_cast<int>(map_width_m_ / map_res_);
        grid_height_ = static_cast<int>(map_height_m_ / map_res_);
        grid_size_ = grid_width_ * grid_height_;
        
        // 使用 float 存储以保持精度，发布时转 int8
        occupancy_grid_.resize(grid_size_, 0.0f); 

        current_speed_ = 0.0f;
        last_update_time_ = ros::Time::now();

        // --- 订阅与发布 ---
        sub_cloud_ = nh.subscribe("/sonar_cloud_body", 1, &SonarOccupancyMapper::cloudCallback, this);
        sub_odom_ = nh.subscribe("/rexrov2/pose_gt", 1, &SonarOccupancyMapper::odomCallback, this);
        pub_map_ = nh.advertise<nav_msgs::OccupancyGrid>("/projected_sonar_map", 1);

        ROS_INFO("Sonar Mapper C++ Started. Grid Size: %dx%d", grid_width_, grid_height_);
    }

private:
    // --- 成员变量 ---
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_odom_;
    ros::Publisher pub_map_;

    std::string map_frame_;
    std::string robot_base_frame_;
    
    float map_res_;
    float map_width_m_, map_height_m_;
    float map_origin_x_, map_origin_y_;
    int grid_width_, grid_height_, grid_size_;
    
    float sonar_max_range_;
    float sonar_fov_rad_;
    float hit_increment_;
    float max_occupancy_;
    float base_decay_rate_;
    float velocity_decay_factor_;
    int dilation_radius_;
    float max_tilt_deg_;

    std::vector<float> occupancy_grid_;
    float current_speed_;
    ros::Time last_update_time_;

    // --- 回调函数 ---
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        float vx = msg->twist.twist.linear.x;
        float vy = msg->twist.twist.linear.y;
        current_speed_ = std::sqrt(vx*vx + vy*vy);
    }

    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        ros::Time current_time = ros::Time::now();
        double dt = (current_time - last_update_time_).toSec();
        if (dt < 0) dt = 0.0;
        last_update_time_ = current_time;

        // 1. 严格姿态检查 (Base Link Tilt Check)
        try {
            geometry_msgs::TransformStamped base_trans = tf_buffer_.lookupTransform(
                map_frame_, robot_base_frame_, cloud_msg->header.stamp, ros::Duration(0.1));
            
            tf2::Quaternion q(
                base_trans.transform.rotation.x,
                base_trans.transform.rotation.y,
                base_trans.transform.rotation.z,
                base_trans.transform.rotation.w);
            
            double roll, pitch, yaw;
            tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

            double max_tilt_rad = max_tilt_deg_ * M_PI / 180.0;
            if (std::abs(roll) > max_tilt_rad || std::abs(pitch) > max_tilt_rad) {
                publishMap(cloud_msg->header.stamp); // 仅发布旧地图
                return;
            }
        } catch (tf2::TransformException &ex) {
            return;
        }

        // 2. 获取传感器位姿
        geometry_msgs::TransformStamped sensor_trans_msg;
        try {
            sensor_trans_msg = tf_buffer_.lookupTransform(
                map_frame_, cloud_msg->header.frame_id, cloud_msg->header.stamp, ros::Duration(0.1));
        } catch (tf2::TransformException &ex) {
            return;
        }

        // 提取传感器位置和 Yaw (用于 FOV 计算)
        float sensor_x = sensor_trans_msg.transform.translation.x;
        float sensor_y = sensor_trans_msg.transform.translation.y;
        
        tf2::Quaternion qs(
            sensor_trans_msg.transform.rotation.x,
            sensor_trans_msg.transform.rotation.y,
            sensor_trans_msg.transform.rotation.z,
            sensor_trans_msg.transform.rotation.w);
        double s_roll, s_pitch, s_yaw;
        tf2::Matrix3x3(qs).getRPY(s_roll, s_pitch, s_yaw);

        // 3. 处理点云 (Hit Update)
        // 将 ROS 消息转换为 PCL 格式
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*cloud_msg, pcl_cloud);

        // 转换点云到 Map Frame
        pcl::PointCloud<pcl::PointXYZ> cloud_world;
        Eigen::Isometry3d transform_eigen = tf2::transformToEigen(sensor_trans_msg);
        // pcl_ros::transformPointCloud 使用 Eigen::Matrix4f，需要转换
        pcl::transformPointCloud(pcl_cloud, cloud_world, transform_eigen.matrix().cast<float>());

        // 创建当前帧的 Hit Mask (避免同一帧对同一栅格重复计数)
        // 使用 vector<bool> 的特化版本或 vector<uint8_t>。uint8_t 更快避免位操作开销。
        std::vector<uint8_t> current_hits(grid_size_, 0);
        bool has_hits = false;

        for (const auto& pt : cloud_world.points) {
            if (std::isnan(pt.x) || std::isnan(pt.y)) continue;

            int cx = static_cast<int>((pt.x - map_origin_x_) / map_res_);
            int cy = static_cast<int>((pt.y - map_origin_y_) / map_res_);

            if (isValid(cx, cy)) {
                current_hits[cy * grid_width_ + cx] = 1;
                has_hits = true;
            }
        }

        if (has_hits) {
            applyHitsWithDilation(current_hits);
        }

        // 4. 应用 FOV 衰减 (Decay Update)
        applyFOVDecay(dt, sensor_x, sensor_y, static_cast<float>(s_yaw));

        // 5. 发布地图
        publishMap(cloud_msg->header.stamp);
    }

    // --- 核心算法实现 ---

    void applyHitsWithDilation(const std::vector<uint8_t>& hits) {
        // 如果没有膨胀，直接更新
        if (dilation_radius_ <= 0) {
            for (int i = 0; i < grid_size_; ++i) {
                if (hits[i]) {
                    occupancy_grid_[i] = std::min(occupancy_grid_[i] + hit_increment_, max_occupancy_);
                }
            }
            return;
        }

        // 带膨胀的更新
        // 为了性能，我们不创建完整的 dilated mask，而是遍历 hits，直接更新周围的 grid
        // 注意：这可能导致重叠区域被多次加分？原 Python 代码逻辑是先 dilate mask (bool OR) 再加分
        // 所以我们需要第二张 mask 或 set 来记录这一帧哪里被“膨胀后”击中了。
        
        std::vector<uint8_t> dilated_hits(grid_size_, 0);

        // 寻找所有 hit 点并标记膨胀区域
        // 遍历整个 hits 数组可能较慢，但比 Python 快。
        // 优化：我们可以在上面的点云循环中记录 hit 的索引，这里只遍历索引。
        // 但为了保持结构简单且 C++ 足够快，这里直接遍历 hits。
        
        for (int y = 0; y < grid_height_; ++y) {
            for (int x = 0; x < grid_width_; ++x) {
                if (hits[y * grid_width_ + x]) {
                    // 膨胀
                    for (int dy = -dilation_radius_; dy <= dilation_radius_; ++dy) {
                        for (int dx = -dilation_radius_; dx <= dilation_radius_; ++dx) {
                            int nx = x + dx;
                            int ny = y + dy;
                            if (isValid(nx, ny)) {
                                dilated_hits[ny * grid_width_ + nx] = 1;
                            }
                        }
                    }
                }
            }
        }

        // 应用增量
        for (int i = 0; i < grid_size_; ++i) {
            if (dilated_hits[i]) {
                occupancy_grid_[i] = std::min(occupancy_grid_[i] + hit_increment_, max_occupancy_);
            }
        }
    }

    void applyFOVDecay(double dt, float sensor_x, float sensor_y, float sensor_yaw) {
        if (dt <= 0) return;

        int range_indices = static_cast<int>(sonar_max_range_ / map_res_) + 1;
        int cx = static_cast<int>((sensor_x - map_origin_x_) / map_res_);
        int cy = static_cast<int>((sensor_y - map_origin_y_) / map_res_);

        int min_x = std::max(0, cx - range_indices);
        int max_x = std::min(grid_width_, cx + range_indices);
        int min_y = std::max(0, cy - range_indices);
        int max_y = std::min(grid_height_, cy + range_indices);

        if (min_x >= max_x || min_y >= max_y) return;

        float decay_amount = (base_decay_rate_ + current_speed_ * velocity_decay_factor_) * static_cast<float>(dt);
        float decay_max_range_sq = std::pow(sonar_max_range_ + 1.0f, 2);
        float half_fov = (sonar_fov_rad_ + (5.0f * M_PI / 180.0f)) / 2.0f;

        for (int y = min_y; y < max_y; ++y) {
            // 预计算 y 轴的世界坐标部分
            float world_y = y * map_res_ + map_origin_y_;
            float dy = world_y - sensor_y;

            for (int x = min_x; x < max_x; ++x) {
                float world_x = x * map_res_ + map_origin_x_;
                float dx = world_x - sensor_x;

                float dist_sq = dx * dx + dy * dy;

                if (dist_sq < decay_max_range_sq) {
                    float angle = std::atan2(dy, dx);
                    float delta_angle = angle - sensor_yaw;
                    
                    // 归一化角度到 -PI ~ PI
                    while (delta_angle <= -M_PI) delta_angle += 2 * M_PI;
                    while (delta_angle > M_PI) delta_angle -= 2 * M_PI;

                    if (std::abs(delta_angle) < half_fov) {
                        int idx = y * grid_width_ + x;
                        occupancy_grid_[idx] = std::max(0.0f, occupancy_grid_[idx] - decay_amount);
                    }
                }
            }
        }
    }

    void publishMap(ros::Time stamp) {
        nav_msgs::OccupancyGrid grid_msg;
        grid_msg.header.stamp = stamp;
        grid_msg.header.frame_id = map_frame_;
        grid_msg.info.resolution = map_res_;
        grid_msg.info.width = grid_width_;
        grid_msg.info.height = grid_height_;
        grid_msg.info.origin.position.x = map_origin_x_;
        grid_msg.info.origin.position.y = map_origin_y_;
        grid_msg.info.origin.position.z = -76.0;
        grid_msg.info.origin.orientation.w = 1.0;

        grid_msg.data.resize(grid_size_);
        
        // 转换 float -> int8
        for (int i = 0; i < grid_size_; ++i) {
            grid_msg.data[i] = static_cast<int8_t>(occupancy_grid_[i]);
        }

        pub_map_.publish(grid_msg);
    }

    inline bool isValid(int x, int y) {
        return (x >= 0 && x < grid_width_ && y >= 0 && y < grid_height_);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sonar_occupancy_mapper");
    SonarOccupancyMapper node;
    ros::spin();
    return 0;
}