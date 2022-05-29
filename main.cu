#include "device.cuh"
// #include <memory>

int main()
{
    loguru::g_stderr_verbosity = 9; // print everything

    DataSet<float> dt("/home/u20/dataset/paper/f3_long_office");
    ck(cudaGetLastError());

    cv::viz::Viz3d viz1("viz1"), viz2("viz2");
    viz1.showWidget("Coordinate", cv::viz::WCoordinateSystem());

    TSDF tsdf(make_uint3(512, 512, 512), make_int2(640, 480));
    tsdf.pintr = new Intr(make_float4(550, 550, 320, 240));
    for (int i = 1; i < dt.pose.frames; i++)
    {
        // i = 1;
        cv::Mat rgb = cv::imread(dt.color_path[i]);
        cv::Mat depth = cv::imread(dt.depth_path[i], cv::IMREAD_ANYDEPTH);
        cv::Affine3f pose = dt.pose.getvectorPose(i);
        tsdf.addScan(depth, rgb,pose);
        // if (i <100)
        //     continue;
        Mat cpu_cloud2;
        Mat cpu_color;
        tsdf.exportCloud(cpu_cloud2, cpu_color);
        // viz1.showWidget("depthmode", cv::viz::WCloud(cpu_cloud2, cpu_color));
        viz1.showWidget("depthmode222", cv::viz::WCloud(cpu_cloud2, cpu_color));
        // Mat depth_color, cpu_cloud,cpu_color;
        // tsdf.depth2cam(depth, rgb, depth_color, cpu_color, cv::Affine3f::Identity());
        // cv::Affine3f viewpose = cv::Affine3f::Identity();
        // viz1.showWidget("depth", cv::viz::WCloud(depth_color, cpu_color), viewpose.translate(cv::Vec3f(4, 0, 0)));

        cv::imshow("rgb", rgb);
        cv::imshow("depth", depth);
        cv::waitKey(10);
        // viz1.spin();
        viz1.spinOnce(true);
    }
}