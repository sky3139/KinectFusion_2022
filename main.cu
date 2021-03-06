#include "TSDF.cuh"
// #include <memory>

int main()
{
    // loguru::g_stderr_verbosity = 9; // print everything

    DataSet<float> dt("/dataset/paper/f3_long_office");
    ck(cudaGetLastError());

    cv::viz::Viz3d viz1("viz1");
    viz1.showWidget("Coordinate", cv::viz::WCoordinateSystem());

    TSDF tsdf(make_uint3(512, 512, 512), make_int2(640, 480));
    tsdf.pintr = new Intr(make_float4(dt.fx, dt.fy, dt.cx, dt.cy));
    for (int i = 1; i < dt.pose.frames; i++)
    {
        // i = 1;
        cv::Mat rgb = cv::imread(dt.color_path[i]);
        cv::Mat depth = cv::imread(dt.depth_path[i], cv::IMREAD_ANYDEPTH);
        cv::Affine3f pose = dt.pose.getvectorPose(i);
        tsdf.addScan(depth, rgb, pose);
        // if (i <100)
        //     continue;
        Mat cpu_cloud2;
        Mat cpu_color;
        tsdf.exportCloud(cpu_cloud2, cpu_color);
        viz1.showWidget("depthmode222", cv::viz::WCloud(cpu_cloud2, cpu_color));
        // viz1.setViewerPose(pose);

        Mat depth_color, cpu_cloud;
        tsdf.depth2cam(depth, rgb, depth_color, cpu_color, cv::Affine3f::Identity());
        cv::Affine3f viewpose = cv::Affine3f::Identity();
        viz1.showWidget("depth", cv::viz::WCloud(depth_color),pose);//, viewpose.translate(cv::Vec3f(4, 0, 0))

        cv::imshow("rgb", rgb);
        cv::imshow("depth", depth);
        cv::waitKey(10);
        // viz1.spin();
        viz1.spinOnce(true);
    }
}