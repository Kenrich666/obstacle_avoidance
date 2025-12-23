#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <marine_acoustic_msgs/ProjectedSonarImage.h>
#include <vector>
#include <cmath>
#include <algorithm>

class SonarToCloudNode {
public:
    SonarToCloudNode() : nh_("~") {
        // 参数获取
        nh_.param<std::string>("target_frame", target_frame_, "blueview_p900_link");
        nh_.param<float>("fov_deg", fov_deg_, 90.0f);
        nh_.param<float>("min_range", min_range_, 1.0f);
        nh_.param<int>("intensity_threshold", intensity_threshold_, 5);

        // 订阅与发布
        // 使用 ros::TransportHints().tcpNoDelay() 减少网络延迟
        sub_ = nh_.subscribe("/rexrov2/blueview_p900/sonar_image_raw", 1, 
                             &SonarToCloudNode::sonarCallback, this, 
                             ros::TransportHints().tcpNoDelay());
        
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/sonar_cloud_body", 1);

        last_beam_count_ = 0;
        ROS_INFO("Sonar PointCloud C++ Node Started. Frame: %s", target_frame_.c_str());
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;

    // 参数
    std::string target_frame_;
    float fov_deg_;
    float min_range_;
    int intensity_threshold_;

    // 缓存 (Look-up Tables)
    int last_beam_count_;
    std::vector<float> cos_table_;
    std::vector<float> sin_table_;
    
    // 临时缓冲，避免每次回调都重新分配大块内存
    std::vector<float> cloud_data_buffer_; 

    void sonarCallback(const marine_acoustic_msgs::ProjectedSonarImage::ConstPtr& msg) {
        // 1. 基础校验
        if (msg->ranges.empty() || msg->image.data.empty()) return;

        size_t n_ranges = msg->ranges.size();
        size_t total_bytes = msg->image.data.size();

        if (total_bytes % n_ranges != 0) {
            ROS_WARN_THROTTLE(1.0, "Image data size mismatch.");
            return;
        }

        int n_beams = total_bytes / n_ranges;

        // 2. 更新角度缓存 (如果波束数发生变化)
        if (n_beams != last_beam_count_) {
            updateTrigTables(n_beams);
        }

        // 3. 数据处理
        // 预估最大点数以预分配内存 (Worst case: all points valid)
        // 每个点需要 x, y, z (3个float)
        cloud_data_buffer_.clear();
        cloud_data_buffer_.reserve(total_bytes * 3 / 2); // 预留一半作为启发式估计

        const uint8_t* raw_image_ptr = msg->image.data.data();
        const float* ranges_ptr = msg->ranges.data();

        // 核心循环：为了效率，外层循环 Range，内层循环 Beam
        // 这样符合 msg->image.data 的内存布局 (Row-Major)，极大提高缓存命中率
        for (size_t r = 0; r < n_ranges; ++r) {
            float range = ranges_ptr[r];
            
            // 距离过滤
            if (range <= min_range_) continue;

            // 计算该行的起始索引
            size_t row_offset = r * n_beams;

            for (size_t b = 0; b < n_beams; ++b) {
                // 获取强度
                uint8_t intensity = raw_image_ptr[row_offset + b];

                // 强度过滤
                if (intensity > intensity_threshold_) {
                    // 查表计算坐标
                    // ROS 坐标系: X Forward, Y Left, Z Up
                    // 原始逻辑: X = r * cos(theta), Y = r * sin(theta)
                    float x = range * cos_table_[b];
                    float y = range * sin_table_[b];
                    float z = 0.0f;

                    cloud_data_buffer_.push_back(x);
                    cloud_data_buffer_.push_back(y);
                    cloud_data_buffer_.push_back(z);
                    
                    // 如果需要强度字段，可以继续 push_back(intensity) 
                    // 但需修改后续 PointCloud2 的字段定义
                }
            }
        }

        // 4. 构建消息
        if (cloud_data_buffer_.empty()) return;

        sensor_msgs::PointCloud2 pc_msg;
        pc_msg.header = msg->header;
        pc_msg.header.frame_id = target_frame_;
        
        pc_msg.height = 1;
        pc_msg.width = cloud_data_buffer_.size() / 3;
        
        // 定义字段 (XYZ)
        sensor_msgs::PointField pf;
        pf.name = "x"; pf.offset = 0; pf.datatype = sensor_msgs::PointField::FLOAT32; pf.count = 1;
        pc_msg.fields.push_back(pf);
        
        pf.name = "y"; pf.offset = 4; pf.datatype = sensor_msgs::PointField::FLOAT32; pf.count = 1;
        pc_msg.fields.push_back(pf);
        
        pf.name = "z"; pf.offset = 8; pf.datatype = sensor_msgs::PointField::FLOAT32; pf.count = 1;
        pc_msg.fields.push_back(pf);

        pc_msg.is_bigendian = false;
        pc_msg.point_step = 12; // 3 * float (4 bytes)
        pc_msg.row_step = pc_msg.point_step * pc_msg.width;
        pc_msg.is_dense = true;

        // 直接内存拷贝到消息体，比 PointCloud2Iterator 更快
        pc_msg.data.resize(cloud_data_buffer_.size() * sizeof(float));
        memcpy(pc_msg.data.data(), cloud_data_buffer_.data(), pc_msg.data.size());

        pub_.publish(pc_msg);
    }

    void updateTrigTables(int n_beams) {
        cos_table_.resize(n_beams);
        sin_table_.resize(n_beams);

        float fov_rad = fov_deg_ * M_PI / 180.0f;
        // Python: np.linspace(fov_rad/2, -fov_rad/2, n_beams)
        // 生成从左到右的角度
        float step = 0.0f;
        if (n_beams > 1) {
            step = -fov_rad / (n_beams - 1);
        }

        float current_angle = fov_rad / 2.0f;
        
        for (int i = 0; i < n_beams; ++i) {
            cos_table_[i] = std::cos(current_angle);
            sin_table_[i] = std::sin(current_angle);
            current_angle += step;
        }

        last_beam_count_ = n_beams;
        ROS_INFO("Rebuilt trigonometric tables for %d beams.", n_beams);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sonar_to_cloud_node");
    SonarToCloudNode node;
    ros::spin();
    return 0;
}