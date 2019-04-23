
// STL
#include <cstdint>
#include <vector>

// MRPT
#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/opengl/COpenGLScene.h>
#include <mrpt/opengl/CFrustum.h>
#include <mrpt/opengl/CGridPlaneXY.h>
#include <mrpt/opengl/CPointCloudColoured.h>
#include <mrpt/opengl/COpenGLViewport.h>
#include <mrpt/opengl/stock_objects.h>

#include <mrpt/math/CMatrixFixedNumeric.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/utils/CImage.h>

// Eigen
#include <Eigen/Core>

// GLog
#include <glog/logging.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "util/macros.h"

// TODO:
//       Add support for lines (mrpt::opengl::CSetOfLines)
//       Add support to remove objects

namespace Eigen {
    typedef Matrix<std::uint8_t, 3, 1> Vector3u;
}

// Color ordering: BGR
class MRPTScene {
public:

    static const Eigen::Vector3u COLOR_RED;
    static const Eigen::Vector3u COLOR_GREEN;
    static const Eigen::Vector3u COLOR_BLUE;
    static const Eigen::Vector3u COLOR_BLACK;
    static const Eigen::Vector3u COLOR_WHITE;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MRPTScene()
        : win("MRPT Scene", 1920, 1080), point_cloud(mrpt::opengl::CPointCloudColoured::Create()) {

        Eigen::Matrix3d R;
        R <<  0, -1,  0,
              0,  0, -1,
              1,  0,  0;

        correction.linear() = R;
        correction.translation() = Eigen::Vector3d::Zero();

        reset();
    }

    virtual ~MRPTScene() { }

    inline std::size_t addImageView(int x, int y, int width, int height) {
        mrpt::opengl::COpenGLScenePtr scene = win.get3DSceneAndLock();

        mrpt::opengl::COpenGLViewportPtr viewport = scene->createViewport(std::string("image") + std::to_string(viewports.size()));
        viewport->setViewportPosition(x, y, width, height);

        viewports.push_back(viewport);

        win.unlockAccess3DScene();
        return viewports.size()-1;
    }

    inline std::size_t addPoint(const Eigen::Vector3d& p, const Eigen::Vector3u& color = COLOR_RED) {
        point_cloud->push_back(p(0), p(1), p(2), color(2), color(1), color(0));
        return point_cloud->size()-1;
    }

    inline std::size_t addPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::Vector3u& color = COLOR_RED) {
        mrpt::opengl::CFrustumPtr frustum = mrpt::opengl::CFrustum::Create();

        Eigen::Isometry3d T;
        T.linear() = R;
        T.translation() = t;

        T = T * correction;

        mrpt::math::CMatrixDouble44 m(T.matrix());

        frustum->setPose(mrpt::poses::CPose3D(m));
        frustum->setColor_u8(color(2), color(1), color(0));
        frustum->setScale(0.1f);

        mrpt::opengl::COpenGLScenePtr scene = win.get3DSceneAndLock();
        scene->insert(frustum);
        win.unlockAccess3DScene();

        frustums.push_back(frustum);
        return frustums.size()-1;
    }

    inline void clear() {
        mrpt::opengl::COpenGLScenePtr scene = win.get3DSceneAndLock();
        scene->clear();
        win.unlockAccess3DScene();

        point_cloud->clear();
        frustums.clear();
        viewports.clear();

        reset();
    }

    inline bool isOpen() { return win.isOpen(); }

    inline void repaint() { win.repaint(); }

    inline void updateImageView(const std::size_t id, cv::Mat image) {
        CHECK_LT(id, viewports.size());

        double x, y, width, height;
        mrpt::opengl::COpenGLViewportPtr viewport = viewports[id];
        viewport->getViewportPosition(x, y, width, height);

        cv::Mat cv_img;
        cv::Size img_sz(width, height);
        cv::resize(image, cv_img, img_sz);

        RUNTIME_ASSERT(cv_img.channels() == 3 || cv_img.channels() == 1);

        bool color = false;
        if (cv_img.channels() == 3) {
            color = true;
            cv_img.convertTo(cv_img, CV_8UC3);
        } else if (cv_img.channels() == 1) {
            color = false;
            cv_img.convertTo(cv_img, CV_8UC1);
        }

        mrpt::utils::CImage mrpt_image;
        mrpt_image.loadFromMemoryBuffer(img_sz.width, img_sz.height, color, cv_img.data, false);
        viewport->setImageView_fast(mrpt_image);
    }

    inline void updatePoint(const std::size_t id, const Eigen::Vector3d& p) {
        CHECK_LT(id, point_cloud->size());

        point_cloud->setPoint_fast(id, p(0), p(1), p(2));
    }

    inline void updatePoint(const std::size_t id, const Eigen::Vector3d& p, const Eigen::Vector3u& color) {
        CHECK_LT(id, point_cloud->size());

        point_cloud->setPoint_fast(id, p(0), p(1), p(2));
        point_cloud->setPointColor_fast(id, color(2) / 255.f, color(1) / 255.f, color(0) / 255.f);
    }

    inline void updatePointColor(std::size_t id, const Eigen::Vector3u& color) {
        CHECK_LT(id, point_cloud->size());

        point_cloud->setPointColor_fast(id, color(2) / 255.f, color(1) / 255.f, color(0) / 255.f);
    }

    inline void updatePose(const std::size_t id, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
        CHECK_LT(id, frustums.size());

        Eigen::Isometry3d T;
        T.linear() = R;
        T.translation() = t;

        T = T * correction;

        mrpt::math::CMatrixDouble44 m(T.matrix());

        frustums[id]->setPose(mrpt::poses::CPose3D(m));
    }

    inline void updatePose(const std::size_t id, const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::Vector3u& color) {
        CHECK_LT(id, frustums.size());

        Eigen::Isometry3d T;
        T.linear() = R;
        T.translation() = t;

        T = T * correction;

        mrpt::math::CMatrixDouble44 m(T.matrix());

        frustums[id]->setPose(mrpt::poses::CPose3D(m));
        frustums[id]->setColor_u8(color(2), color(1), color(0));
    }

    inline void updatePoseColor(const std::size_t id, const Eigen::Vector3u& color) {
        CHECK_LT(id, frustums.size());

        frustums[id]->setColor_u8(color(2), color(1), color(0));
    }

    inline int waitForKey() { return win.waitForKey(); }

private:

    mrpt::gui::CDisplayWindow3D win;
    mrpt::opengl::CPointCloudColouredPtr point_cloud;
    std::vector<mrpt::opengl::CFrustumPtr> frustums;
    std::vector<mrpt::opengl::COpenGLViewportPtr> viewports;

    Eigen::Isometry3d correction;

    inline void reset() {
        point_cloud->setPointSize(3.f);

        mrpt::opengl::COpenGLScenePtr scene = win.get3DSceneAndLock();
        scene->insert(mrpt::opengl::CGridPlaneXY::Create(-10, 10, -10, 10));
        scene->insert(mrpt::opengl::stock_objects::CornerXYZSimple());
        scene->insert(point_cloud);
        win.unlockAccess3DScene();
    }
};

const Eigen::Vector3u MRPTScene::COLOR_RED   = Eigen::Vector3u(  0,   0, 255);
const Eigen::Vector3u MRPTScene::COLOR_GREEN = Eigen::Vector3u(  0, 255,   0);
const Eigen::Vector3u MRPTScene::COLOR_BLUE  = Eigen::Vector3u(255,   0,   0);
const Eigen::Vector3u MRPTScene::COLOR_BLACK = Eigen::Vector3u(  0,   0,   0);
const Eigen::Vector3u MRPTScene::COLOR_WHITE = Eigen::Vector3u(255, 255, 255);
