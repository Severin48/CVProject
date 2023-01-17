#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>

using namespace cv;
using namespace std;

const int defaultWidth = 640;
const int defaultHeight = 480;
const float defaultRatio = (float)defaultWidth / defaultHeight;

map<char, char> fileMap = { {0, 'a'}, {1, 'b'}, {2, 'c'}, {3, 'd'}, {4, 'e'}, {5, 'f'}, {6, 'g'}, {7, 'h'} };


const std::string w_name = "Live";

const float max_angle_deviation = 0.01;


class Board {
public:
    std::vector<cv::Point> corners;
    cv::Rect boardRect;
    float sideLength = 0;
    bool located = false;
    cv::Rect roi;

    Board() {
    }

    Board(std::vector<cv::Point> corners, cv::Rect roi) {
        this->located = true;
        this->roi = roi;
        this->sideLength = roi.width;
    }
};

class Piece {
public:
    string name;
    string squareName;
    Point squareTopLeft;
    bool active;
    bool isWhite;

    Piece() {}

    Piece(string squareName, Point squareTopLeft, bool isWhite) {
        this->squareName = squareName;
        this->squareTopLeft = squareTopLeft;
        this->isWhite = isWhite;
    }
};

class Square {
public:
    Point topLeft;
    bool occupied = false;
    Piece piece;
    string name;
    Rect rect;
    char file;
    char rank;
    bool isWhite;
    Scalar meanCol;
    Mat img;
    int meanSum;
    bool occupiedByWhite;

    Square() {}

    Square(Point tl, Rect rec, char file, char rank, Scalar meanCol, Mat img, bool isWhite, int meanSum) { // File: Letters, Rank: Numbers
        this->topLeft = tl;
        this->rect = rec;
        this->file = file;
        this->rank = rank;
        this->name = string() + file + rank;
        this->meanCol = meanCol;
        this->img = img;
        this->isWhite = isWhite;
        this->meanSum = meanSum;
    }
};

class Game {
public:
    Board board;
    vector<Square> squares;
    vector<Piece> pieces;

    Game() {

    }

    Game(Board b) {
        this->board = b;
    }
};

double euclideanDist(cv::Point p1, cv::Point p2) {
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

double vecLen(Vec4i vec) {
    return euclideanDist(Point(vec[0], vec[1]), Point(vec[2], vec[3]));
}

double vecLen(Point p) {
    return euclideanDist(p, Point(0, 0));
}

std::pair<Point, Point> pointsFromVec4i(cv::Vec4i line) {
    return { Point(line[0], line[1]), Point(line[2], line[3]) };
}

Vec4i unitVec(Vec4i vec) {
    std::pair<Point, Point> points = pointsFromVec4i(vec);
    Point p1 = points.first;
    Point p2 = points.second;
    double length = vecLen(vec);
    Vec4i unitvec = vec / length;
    return unitvec;
}

double distBetweenLines(Vec4i pVec, Vec4i qVec) {
    Point p1 = Point(pVec[0], pVec[1]);
    Point p2 = Point(pVec[2], pVec[3]);
    Point q = Point(qVec[0], qVec[1]);
    return vecLen((q - p1) - ((q - p1).dot(p2 - p1) / pow(euclideanDist(p1, p2), 2)) * (p2 - p1));
}

double getMaxDistLines(std::vector<Vec4i> lines, std::pair<Vec4i, Vec4i>& farthestLines) {
    // Lines are parallel within the group --> Only one unit vec needed
    double maxDist = 0;
    double dist = 0;
    for (Vec4i outerLine : lines) {
        //Vec4i unVec = unitVec(outerLine);
        for (Vec4i innerLine : lines) {
            dist = distBetweenLines(outerLine, innerLine);
            if (dist > maxDist) {
                maxDist = dist;
                farthestLines.first = outerLine;
                farthestLines.second = innerLine;
            }
        }
    }
    cout << "Max dist between lines: " << maxDist << endl;
    return maxDist;
}

cv::Point midpoint(const cv::Point& a, const cv::Point& b) {
    cv::Point ret;
    ret.x = (a.x + b.x) / 2.;
    ret.y = (a.y + b.y) / 2.;
    return ret;
}

cv::Point midpoint(const cv::Vec4i& v) {
    cv::Point ret;
    ret.x = (v[0] + v[2]) / 2.;
    ret.y = (v[1] + v[3]) / 2.;
    return ret;
}

template <typename T>
bool in_vec_within_tolerance(T testValue, std::vector<T>& vec, float allowedDeviation, int& idx) {
    // TODO: Test with Point class members
    for (int i = 0; i < vec.size(); i++) {
        if (fabs(vec[i] - testValue) < allowedDeviation) {
            idx = i;
            return true;
        }
    }
    return false;
}


template <typename T>
void printVec(std::string s, std::vector<T>& vec) {
    std::cout << s << std::endl;
    for (auto i : vec)
        std::cout << i << ' ' << std::endl;
    //std::cout << std::endl;
}


template <typename T>
bool within_tolerance(T val1, T val2, float allowedDeviation) {
    // TODO: Test with Point class members
    if (fabs(val2 - val1) < allowedDeviation) {
        return true;
    }
    return false;
}


double median(std::vector<double>& v) {
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}


cv::Mat getFrame(cv::VideoCapture& cap) {
    cv::Mat ret;
    while (ret.empty()) cap >> ret;
    //std::cout << ret.cols << ", " << ret.rows << " " << std::endl;
    float newRatio = (float)ret.cols / ret.rows;
    if (cv::Size(ret.cols, ret.rows) != cv::Size(defaultWidth, defaultHeight)) {
        resize(ret, ret, cv::Size(defaultWidth, defaultWidth / newRatio));
    }
    //std::cout << defaultWidth << ", " << defaultWidth / newRatio << " " << std::endl;
    return ret;
}

cv::Mat rotate_image(cv::Mat& src, double angle) {
    std::cout << "\nRotating image..." << std::endl;
    angle = angle * (180 / CV_PI);
    std::cout << "Rotation angle (deg): " << angle << std::endl;
    cv::Point center = cv::Point(src.cols / 2, src.rows / 2);
    cv::Mat rot_mat = getRotationMatrix2D(center, angle, 1.);
    cv::Mat rotate_dst;
    warpAffine(src, rotate_dst, rot_mat, src.size());

    return rotate_dst;
}

cv::Mat rotate_image(cv::Mat& src, double angle, vector<Point>& corners) {
    std::cout << "\nRotating image..." << std::endl;
    angle = angle * (180 / CV_PI);
    std::cout << "Rotation angle (deg): " << angle << std::endl;
    cv::Point center = cv::Point(src.cols / 2, src.rows / 2);
    cv::Mat rot_mat = getRotationMatrix2D(center, angle, 1.);
    cv::Mat rotate_dst;
    warpAffine(src, rotate_dst, rot_mat, src.size());
    //imshow("Rotated", rotate_dst);

    std::vector<Point3d> newCornersRot;
    for (Point p : corners) {
        newCornersRot.push_back(Point3d(p.x, p.y, 1));
    }
    cv::Mat pDst;
    for (int i = 0; i < newCornersRot.size(); i++)
    {
        pDst = (rot_mat * Mat(newCornersRot[i])).t();

        corners[i] = Point(pDst);
        //cout << newCornersRot[i] << " --> " << pDst << "\n";

    }
    return rotate_dst;
}

double getAngle(cv::Point p1, cv::Point p2) {
    double angle;
    if (p2.x - p1.x >= DBL_EPSILON) angle = atan2(p2.y - p1.y, p2.x - p1.x);
    else angle = CV_PI / 2.;
    //std::cout << std::endl << "Before - Angle in degrees:\t" << angle * 180 / CV_PI << std::endl;
    angle = angle <= -CV_PI / 2. ? angle + CV_PI : angle;
    angle = angle >= CV_PI / 2. ? angle - CV_PI : angle;
    return angle;
}

cv::Point getIntersectionOfExtendedLines(cv::Vec4i l1, cv::Vec4i l2) {
    cv::Point p11 = cv::Point(l1[0], l1[1]);
    cv::Point p12 = cv::Point(l1[2], l1[3]);
    cv::Point p21 = cv::Point(l2[0], l2[1]);
    cv::Point p22 = cv::Point(l2[2], l2[3]);

    // y = m*x + c
    double m1, m2;
    if (p12.x - p11.x == 0) m1 = DBL_MAX; // TODO: Nicht DBL_MAX - Gefahr von Überlauf sondern einfach p11.x, p21.y(p11.x) returnen, analog bei m2
    else m1 = ((float)p12.y - p11.y) / (p12.x - p11.x);
    if (p22.x - p21.x == 0) m2 = DBL_MAX;
    else m2 = ((float)p22.y - p21.y) / (p22.x - p21.x);

    double c1, c2;
    c1 = p11.y - m1 * p11.x;
    c2 = p21.y - m2 * p21.x;

    double x_i, y_i;
    // m1*x + c1 - y = m2*x + c2 - y --> m1x1 - m2x2 = c2 - y2 + y1 - c1 (da x1=x2, y1=y2) --> x_i = (c2-c1) / (m1-m2);
    x_i = (c2 - c1) / (m1 - m2);
    y_i = m1 * x_i + c1;
    return cv::Point((int)x_i, (int)y_i);
}

void onClickLive(int event, int x, int y, int z, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        std::cout << "x=" << x << ", y=" << y << std::endl;
    }
}

void onClickLines(int event, int x, int y, int z, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        std::cout << "x=" << x << ", y=" << y << std::endl;
    }
}

cv::VideoCapture select_camera() {
    cv::Mat frame;
    cv::VideoCapture cap;

    int deviceID = 1;       // USB secondary CAM for Laptop 
    int defaultID = 0;      // 0 = open default camera

    cap.open(deviceID);
    bool opened = false;
    opened = cap.isOpened();
    cap >> frame;
    if (frame.empty() || !opened || frame.data >= frame.dataend || frame.datastart == nullptr || frame.data == NULL) {
        cap.open(defaultID);
        opened = cap.isOpened();
        cap >> frame;

        int counter = 0;
        for (uchar* p = frame.data; p != frame.dataend; p++) {
            if (*p == 205 || *p == 0) {
                counter++;
            }
        }
        if (counter > frame.size().width * frame.size().height * frame.channels() / 90) {
            cap.release();
            opened = false;
        }
    }
    else {
        int counter = 0;
        for (uchar* p = frame.data; p != frame.dataend; p++) {
            if (*p == 205 || *p == 0) {
                counter++;
            }
        }
        if (counter > frame.size().width * frame.size().height * frame.channels() / 90) {
            cap.open(defaultID);
            opened = cap.isOpened();
            cap >> frame;
        }
    }

    string videoPath = "C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221213_20_28_03_Pro.mp4";
    string videoPath2 = "C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221213_20_27_35_Pro.mp4";
    string videoPath3 = "C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221214_21_16_23_Pro.mp4";
    string videoPath4 = "C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221214_21_16_49_Pro.mp4";

    cap >> frame;
    Mat grayFrame, binFrame;
    cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    threshold(grayFrame, binFrame, 40, 40, THRESH_BINARY);
    try {
        if (countNonZero(binFrame) < 100 || frame.empty() || !opened) {
            cap.open(defaultID);
        }
    }
    catch (std::exception& e) {
        cap.open(defaultID);
    }
    cap >> frame;
    if (frame.empty() || !opened) {
        std::cerr << "ERROR! Unable to open camera. Using video recording.\n";
        cap.release();
        cap.open(videoPath);
    }
    cap >> frame;
    cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    threshold(grayFrame, binFrame, 40, 40, THRESH_BINARY);
    try {
        if (countNonZero(binFrame) < 100 || frame.empty() || !opened) {
            std::cerr << "ERROR! Unable to open camera. Using video recording.\n";
            cap.release();
            cap.open(videoPath);
        }
        else {
            std::cout << "Nonzero: " << countNonZero(binFrame) << std::endl;
        }
    }
    catch (std::exception& e) {
        std::cerr << "ERROR! Unable to open camera. Using video recording.\n";
        cap.release();
        cap.open(videoPath);
    }
    return cap;
}