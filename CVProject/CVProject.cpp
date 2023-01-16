#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

const int defaultWidth = 640;
const int defaultHeight = 480;
const float defaultRatio = (float)defaultWidth / defaultHeight;


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

        Board(std::vector<cv::Point> corners, float sideLength, cv::Rect roi) {
            //this->boardRect = cv::Rect(corners[0], corners[1], corners[2], corners[3]);
            this->located = true;
            this->roi = roi;
        }
};

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
        resize(ret, ret, cv::Size(defaultWidth, defaultWidth/newRatio));
    }
    //std::cout << defaultWidth << ", " << defaultWidth / newRatio << " " << std::endl;
    return ret;
}

std::vector<Point2f> pointVecToTwofVec(std::vector<Point>& vec) {
    std::vector<Point2f> ret;
    for (Point p : vec) {
        ret.push_back((Point2f)p);
    }
    return ret;
}

std::vector<Point> twofVecToPointVec(std::vector<Point2f>& vec) {
    std::vector<Point> ret;
    for (Point2f p : vec) {
        ret.push_back((Point)p);
    }
    return ret;
}

cv::Mat rotate_image(cv::Mat& src, double angle, vector<Point>& corners) {
    std::cout << "Rotating image..." << std::endl;
    angle = angle * (180 / CV_PI);
    std::cout << "Rotation angle (deg): " << angle << std::endl;
    cv::Point center = cv::Point(src.cols / 2, src.rows / 2);
    cv::Mat rot_mat = getRotationMatrix2D(center, angle, 1.);
    cv::Mat rotate_dst;
    warpAffine(src, rotate_dst, rot_mat, src.size());
    imshow("Rotated", rotate_dst);

    std::vector<Point2f> inPoints = pointVecToTwofVec(corners);
    std::vector<Point2f> outPoints;

    std::cout << "Corners before" << std::endl;
    for (Point corner : corners) {
        cout << corner.x << " " << corner.y << " " << std::endl;
    }
    std::vector<Point3d> newCornersRot;
    for (Point p : corners) {
        newCornersRot.push_back(Point3d(p.x, p.y, 1));
    }
    cv::Mat pDst;
    for (int i = 0; i < newCornersRot.size(); i++)
    {
        pDst = (rot_mat * Mat(newCornersRot[i])).t();

        corners[i] = Point(pDst);
        cout << newCornersRot[i] << " ---> " << pDst << "\n";

    }
    return rotate_dst;
}

double euclideanDist(cv::Point p1, cv::Point p2) {
    double leng;
    return leng = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
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

std::pair<cv::Point, cv::Point> adjustLineLen(cv::Point p1, cv::Point p2, double newLen) {
    // TODO
    return std::pair(p1, p2);
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

//cv::Point subPoints(cv::Point p1, cv::Point p2) {
//    return cv::Point(p1.x-p2.x, p1.y-p2.y);
//}

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


double getSquareLength(cv::VideoCapture& cap, double angle1, double angle2, float meanSideLen, cv::Rect roi) {
    std::cout << "Determining square length..." << std::endl;
    double boardLen = -1;
    double angleTolerance = 0.1;
    while (boardLen == -1) {
        boardLen = -1;
        std::vector<cv::Mat> frames;
        std::vector<cv::Vec4i> all_lines, relevantLines;
        int nFrames = 1;
        frames.reserve(nFrames);
        cv::Mat frame, linesImg;
        for (int i = 0; i < nFrames; i++) {
            frame = getFrame(cap);
            cv::Mat image_roi = frame(roi);
            if (frame.empty()) continue;
            linesImg = image_roi.clone();
            frames.push_back(image_roi);
            cv::Mat grayImg;
            cvtColor(image_roi, grayImg, cv::COLOR_BGR2GRAY);
            //imshow("Gray", grayImg);

            // TODO: Vordergrund und Hintergrund --> Alles auf 0 setzen, was außerhalb der Corners ist.

            cv::Mat blurGray, edges;
            GaussianBlur(grayImg, blurGray, cv::Size(5, 5), 0);
            //imshow("Blur gray", blurGray);

            Canny(blurGray, edges, 50, 150);
            //imshow("Edges Square", edges);

            // https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv --> Auch für Felder nützlich

            std::vector<cv::Vec4i> lines;
            // HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 10, 20); // --> Really good, but diagonals in squares

            // HoughLinesP(edges, lines, 1, CV_PI / 180, 100, 10, 30); --> Very good for the board

            // HoughLinesP(edges, lines, 1, CV_PI / 180, 100, 150, 30); // Also very good for the board

            // HoughLinesP(edges, lines, 1, CV_PI / 180., 100, 150, 20); // --> Perfect for the board, sometimes not fully recognized --> Use multiple frames and compare results

            HoughLinesP(edges, lines, 1, CV_PI / 180., 40, 10, 4); // TODO: TWEAK + Auf Bildgröße anpassen, nicht statisch

            all_lines.insert(all_lines.end(), lines.begin(), lines.end());
        }

        if (all_lines.size() < 2) {
            continue; // Potentiell endlosschleife
        }

        double angle;
        std::vector<double> lengths;
        for (cv::Vec4i l : all_lines) {
            cv::Point p1, p2;
            p1 = cv::Point(l[0], l[1]);
            p2 = cv::Point(l[2], l[3]);
            angle = getAngle(p1, p2);
            if (within_tolerance(angle, angle1, angleTolerance) || within_tolerance(angle, angle2, angleTolerance)) {
                relevantLines.push_back(l);
                line(linesImg, p1, p2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                lengths.push_back(euclideanDist(p1, p2));
            }

        }

        double medianLen = median(lengths);

        std::cout << "Median square length: " << medianLen << std::endl;

        imshow("LinesSquare", linesImg);

        frames.clear();

        boardLen = medianLen;
    }

    return boardLen;
}

Board getBoard(cv::VideoCapture& cap) {
    std::cout << std::endl << "Starting board detection..." << std::endl;
    cv::Mat frame, dst;
    cv::Mat linesImg;
    //cap >> linesImg;
    linesImg = getFrame(cap);
    //resize(linesImg, linesImg, cv::Size(640, 480)); // TODO: Falls resize verwendet wird, Ratio einlesen und dann anpassen. Aber sollte nicht verwendet werden, da es dann überall gemacht
    // werden müsste
    std::vector<cv::Mat> boardFrames;
    int n_boardFrames = 20;
    boardFrames.reserve(n_boardFrames);
    float allowedAngleDeviationRad = 0.1;

    std::vector<float> angles;
    std::vector<std::vector<int>> groupedIndices;
    std::vector<cv::Vec4i> all_lines;
    int angle_counter = 0;
    for (int i = 0; i < n_boardFrames; i++) {
        //cap >> frame;
        frame = getFrame(cap);
        if (frame.empty()) continue;
        dst = frame.clone();
        boardFrames.push_back(frame);

        //std::cout << "New image" << std::endl;
        cv::Mat grayImg;
        cvtColor(frame, grayImg, cv::COLOR_BGR2GRAY);
        //imshow("Gray", grayImg);

        cv::Mat blurGray, edges;
        GaussianBlur(grayImg, blurGray, cv::Size(5, 5), 0);
        //imshow("Blur gray", blurGray);

        Canny(blurGray, edges, 50, 150);
        //imshow("Edges", edges);

        // https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv --> Auch für Felder nützlich

        std::vector<cv::Vec4i> lines;
        // HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 10, 20); // --> Really good, but diagonals in squares

        // HoughLinesP(edges, lines, 1, CV_PI / 180, 100, 10, 30); --> Very good for the board

        // HoughLinesP(edges, lines, 1, CV_PI / 180, 100, 150, 30); // Also very good for the board

        // HoughLinesP(edges, lines, 1, CV_PI / 180., 100, 150, 20); // --> Perfect for the board, sometimes not fully recognized --> Use multiple frames and compare results

        // HoughLinesP(edges, lines, 1, CV_PI / 180., 100, 150, 20); // --> Main variant - really good for board detection but missing inner lines

        HoughLinesP(edges, lines, 1, CV_PI / 180., 100, 150, 20);
        
        // TODO: Auf Bildgröße anpassen, nicht statisch

        all_lines.insert(all_lines.end(), lines.begin(), lines.end());

        // https://stackoverflow.com/questions/15888180/calculating-the-angle-between-points
        groupedIndices.resize(angle_counter + lines.size());
        for (size_t i = 0; i < lines.size(); i++) {
            cv::Vec4i l = lines[i];

            cv::Point p1, p2;
            p1 = cv::Point(l[0], l[1]);
            p2 = cv::Point(l[2], l[3]);
            double angle = getAngle(p1, p2); // Calculate angle in radian,  if you need it in degrees just do angle * 180 / PI

            //std::cout << "After - Angle in degrees:\t" << angle * 180 / CV_PI << std::endl;

            line(linesImg, p1, p2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

            //double leng = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
            double leng = euclideanDist(p1, p2);

            //std::cout << "p1\tx=" << p1.x << ", y=" << p1.y << std::endl;
            //std::cout << "p2\tx=" << p2.x << ", y=" << p2.y << std::endl;
            //std::cout << i << "\tangle=" << angle << "\teuclidean_length=" << leng << std::endl;

            int idx = 0;
            if (in_vec_within_tolerance<float>(angle, angles, allowedAngleDeviationRad, idx)) {
                groupedIndices[idx].push_back(angle_counter);
            }
            else {
                groupedIndices[angle_counter].push_back(angle_counter);
            }

            angles.push_back(angle); // Adding current angle to vector after checking for similar angles
            angle_counter++;
        }
    }

    std::vector<double> avgAngles;
    avgAngles.resize(groupedIndices.size());
    for (int i = 0; i < groupedIndices.size(); i++) {
        std::vector<int> currentIndices = groupedIndices[i];
        double sum = 0;

        //if (currentIndices.size() > 0) std::cout << std::endl << "Angle group " << i + 1 << std::endl;

        int actualSize = 0;
        double currentVal;
        for (int j = 0; j < currentIndices.size(); j++) {
            currentVal = angles[currentIndices[j]];
            if (currentVal != 0) actualSize++;
            //std::cout << currentVal << std::endl;
            sum += currentVal;
        }
        
        if (actualSize > 1) {
            avgAngles[i] = sum / actualSize;
            std::cout << "Group " << i + 1 << ":\tAverage angle:\t" << std::setprecision(2) << sum / actualSize << "\tLines in this group:\t" << actualSize << std::endl;
        } else avgAngles[i] = -DBL_MAX; // To avoid auto-filling with 0 and thus wrong matching later on.
    }

    // Looking for group-pair with approx. 90°(M_PI / 2) absolute difference.
    if (avgAngles.size() < 2) {
        cout << "Not enough angle groups detected - ending board detection." << endl;
        return Board();
    }
    int maxAppearedIdx = -1;
    int idxGroup1 = -1;
    int idxGroup2 = -1;
    float allowedRightAngleDeviationRad = 0.1;
    std::vector<std::pair<int, int>> possibleGroupPairs;
    int rest = avgAngles.size() % 2;
    for (int i = 0; i < (avgAngles.size() / 2) + rest; i++) {
        for (int j = 1; j < avgAngles.size(); j++) {
            if (i != j) {
                double angle1 = avgAngles[i];
                double angle2 = avgAngles[j];
                if (fabs(fabs(angle1 - angle2) - CV_PI / 2.) < allowedRightAngleDeviationRad && std::max(i, j) > maxAppearedIdx) {
                    maxAppearedIdx = std::max(i, j); // To avoid permutated duplicate groups
                    possibleGroupPairs.push_back(std::pair(i, j));
                    std::cout << "Orthogonal pair found: " << i + 1 << " & " << j + 1 << " with " << std::fixed <<
                        fabs(fabs(angle1 - angle2) - CV_PI / 2.) *180/CV_PI << " degrees deviation from right angle." << std::endl;
                }
            }
        }
    }
    if (possibleGroupPairs.size() == 0) {
        // TODO: (Optional) Andere Parameter müssen berücksichtigt werden. Oder Bild muss so verändert werden, dass perfekte Draufsicht simuliert wird.
        // Siehe CV_6_GeometrischeOperationen.pdf --> Verzerrung
        return Board();
    }

    // Average coordiante of all points in the two groups --> Mittelpunkt des Bretts --> Linien, welche am nähesten an diesem Punkt sind
    // -- > Zwei pro Seite-- > Aber Achtung - dürfen nicht auf der selben Seite liegen-- > gegeneinander Distanz checken muss 2 * die Distanz
    // sein zum Mittelpunkt
    for (std::pair p : possibleGroupPairs) {
        std::vector<cv::Vec4i> boardLimits;
        int total_x = 0;
        int total_y = 0;

        int mean_x = 0;
        int mean_y = 0;
        std::vector<int> g1Indices = groupedIndices[p.first];
        std::vector<int> g2Indices = groupedIndices[p.second];
        std::vector< cv::Vec4i> g1Lines;
        std::vector< cv::Vec4i> g2Lines;
        std::vector<double> distancesToCenterG1;
        std::vector<double> distancesToCenterG2;
        for (auto i : g1Indices) {
            g1Lines.push_back(all_lines[i]);
            cv::Vec4i l = all_lines[i];
            cv::Point p1, p2;
            p1 = cv::Point(l[0], l[1]);
            p2 = cv::Point(l[2], l[3]);
            total_x += p1.x + p2.x;
            total_y += p1.y + p2.y;
        }
        for (auto i : g2Indices) {
            g2Lines.push_back(all_lines[i]);
            cv::Vec4i l = all_lines[i];
            cv::Point p1, p2;
            p1 = cv::Point(l[0], l[1]);
            p2 = cv::Point(l[2], l[3]);
            total_x += p1.x + p2.x;
            total_y += p1.y + p2.y;
        }
        
        int total_size = 2*(g1Indices.size() + g2Indices.size()); // 2* because each Line is defined by 2 points.
        if (total_size == 0) continue;
        mean_x = total_x / total_size;
        mean_y = total_y / total_size;

        cv::Point boardCenter = cv::Point(mean_x, mean_y);

        std::cout << "Board center x = " << mean_x << ", Board center y = " << mean_y << std::endl;
        cv::circle(linesImg, boardCenter, 1, cv::Scalar(0, 255, 0), 2, 8, 0);

        imshow("Lines", linesImg); // TODO: Remove

        // TODO: Get max distance of each group, filter out all lines from groups which are less than 75% of the max_distance
        // Mittelpunkt kann nicht genutzt werden, da er unbalanciert sein kann
        // Für Gruppe 1: Am weitest entfernte zwei Linien finden (diese sind keine Kandidaten) dann alle anderen Linien durchschauen, deren
        // Distanz zu beiden weitesten Linien errechnen und falls die größere dieser beiden Distanzen < 75 % maxDist, dann verwerfe Linie!
        // Dann für Gruppe 2 genau das gleiche

        for (auto l : g1Lines) {
            cv::Point p1 = cv::Point(l[0], l[1]);
            cv::Point p2 = cv::Point(l[2], l[3]);
            distancesToCenterG1.push_back(euclideanDist(midpoint(p1, p2), boardCenter));
            //std::cout << euclideanDist(midpoint(p1, p2), boardCenter) << std::endl;
        }

        for (auto l : g2Lines) {
            cv::Point p1 = cv::Point(l[0], l[1]);
            cv::Point p2 = cv::Point(l[2], l[3]);
            distancesToCenterG2.push_back(euclideanDist(midpoint(p1, p2), boardCenter));
        }

        if (distancesToCenterG1.size() < 2 || distancesToCenterG2.size() < 2) return Board(); // Not enough board limiting lines found.

        int smallestIdx = std::min_element(distancesToCenterG1.begin(), distancesToCenterG1.end()) - distancesToCenterG1.begin();
        double smallest = distancesToCenterG1[smallestIdx];
        double second = DBL_MAX;
        int secondIdx = -1;
        cv::Point midSmallest = midpoint(g1Lines[smallestIdx]);
        for (int i = 0; i < distancesToCenterG1.size(); i++) {
            double distToFirstLine = euclideanDist(midSmallest, midpoint(g1Lines[i]));
            double d = distancesToCenterG1[i];
            if (d <= second && i != smallestIdx && d < distToFirstLine) {
                secondIdx = i;
                second = d;
            }
        }
        
        std::cout << smallestIdx << ": " << smallest << ", " << secondIdx << ": " << second << std::endl;

        if (smallestIdx == -1 || secondIdx == -1) return Board();

        boardLimits.push_back(g1Lines[smallestIdx]);
        boardLimits.push_back(g1Lines[secondIdx]);

        smallestIdx = std::min_element(distancesToCenterG2.begin(), distancesToCenterG2.end()) - distancesToCenterG2.begin();
        smallest = distancesToCenterG2[smallestIdx];
        secondIdx = -1;
        second = DBL_MAX;
        midSmallest = midpoint(g2Lines[smallestIdx]);
        for (int i = 0; i < distancesToCenterG2.size(); i++) {
            double distToFirstLine = euclideanDist(midSmallest, midpoint(g2Lines[i]));
            double d = distancesToCenterG2[i];
            if (d <= second && i != smallestIdx && d < distToFirstLine) {
                secondIdx = i;
                second = d;
            }
        }

        std::cout << smallestIdx << ": " << smallest << ", " << secondIdx << ": " << second << std::endl;

        if (smallestIdx == -1 || secondIdx == -1) return Board();

        boardLimits.push_back(g2Lines[smallestIdx]);
        boardLimits.push_back(g2Lines[secondIdx]);

        std::vector<cv::Point> corners;

        corners.push_back(getIntersectionOfExtendedLines(boardLimits[0], boardLimits[2]));
        std::cout << "Corner 1: " << getIntersectionOfExtendedLines(boardLimits[0], boardLimits[2]) << std::endl;
        corners.push_back(getIntersectionOfExtendedLines(boardLimits[0], boardLimits[3]));
        std::cout << "Corner 2: " << getIntersectionOfExtendedLines(boardLimits[0], boardLimits[3]) << std::endl;
        corners.push_back(getIntersectionOfExtendedLines(boardLimits[1], boardLimits[2]));
        std::cout << "Corner 3: " << getIntersectionOfExtendedLines(boardLimits[1], boardLimits[2]) << std::endl;
        corners.push_back(getIntersectionOfExtendedLines(boardLimits[1], boardLimits[3]));
        std::cout << "Corner 4: " << getIntersectionOfExtendedLines(boardLimits[1], boardLimits[3]) << std::endl;

        for (cv::Point p : corners) {
            if (p.x == INT_MAX || p.y == INT_MAX) return Board();
        }

        float boardSideLen1 = euclideanDist(corners[0], corners[1]);
        float boardSideLen2 = euclideanDist(corners[0], corners[2]);
        float boardSideLen3 = euclideanDist(corners[3], corners[1]);
        float boardSideLen4 = euclideanDist(corners[3], corners[2]);
        float meanSideLen = (boardSideLen1 + boardSideLen2 + boardSideLen3 + boardSideLen4) / 4.;

        double angle1 = avgAngles[p.first];
        std::vector<cv::Point> rotatedCorners(corners); // Create copy of corners
        cv::Mat rotated = rotate_image(frame, angle1, rotatedCorners);
        cv::Point center = cv::Point(frame.cols / 2, frame.rows / 2);

        double angle2 = avgAngles[p.second];
        int minX = INT_MAX;
        int maxY = -1;
        Point topLeft = Point(INT_MAX, INT_MAX);
        Point botRight = Point(0, 0);
        int tolerancePxs = 20;
        for (Point p : rotatedCorners) {
            if (p.x < minX && p.y < topLeft.y - tolerancePxs) {
                minX = p.x;
                topLeft = p;
            }
        }
        for (Point p : rotatedCorners) {
            if (p.y > maxY && p.x > botRight.x + tolerancePxs) {
                maxY = p.y;
                botRight = p;
            }
        }
        if (minX == INT_MAX || maxY == -1) { return Board(); }
        cv::Rect roi = cv::Rect(topLeft, botRight);
        Mat rotatedCpy = rotated.clone();
        cv::rectangle(rotatedCpy, roi, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        imshow("Rotated roi", rotatedCpy);
        double squareLen = getSquareLength(cap, angle1, angle2, meanSideLen, roi);

        std::pair<cv::Point, cv::Point> correctedLine1 = adjustLineLen(rotatedCorners[0], rotatedCorners[1], squareLen);

        line(dst, corners[0], corners[1], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        line(dst, corners[0], corners[2], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        line(dst, corners[3], corners[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        line(dst, corners[3], corners[2], cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

        line(rotated, rotatedCorners[0], rotatedCorners[1], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        line(rotated, rotatedCorners[0], rotatedCorners[2], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        line(rotated, rotatedCorners[3], rotatedCorners[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        line(rotated, rotatedCorners[3], rotatedCorners[2], cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

        imshow("Rotated", rotated);
        imshow(w_name, dst);

        return Board(corners, 8*squareLen, roi);

        // TODO: Gruppierung in Funktion mostPrevalentGroups ausbauen --> Kann dann auch für Gruppen von Feldgrößen genutzt werden --> Kann bei der Erkennung der Brettbegrenzung helfen
        // Insbesondere bei der Größe der Begrenzungslinien! (Ca. 8*Feldgröße = Brettgröße), aber vorsicht, boardCenter liegt nicht genau in der Mitte.

        // TODO: Smallest --> Second smallest under condition that it is on the other side --> Compare distances if take smallest of G1 and G2, move farther line
        // such that it is 

        // Analog für G2

        // TODO: Beide Gruppen durchgehen, bei beiden gleiches Vorgehen, um beide Innenlinien zu ermitteln:
        // Linien nach aufsteigendem Abstand zum Mittelpunkt sortieren. Erste Linie ist die mit kleinstem Abstand. Zweite Linie ist die mit nächstkleinem Abstand, wobei allerdings
        // auch gelten muss, dass die erste Linie nicht zwischen dieser Linie und dieser Kandidatenlinie liegt. Mathematisch ausgedrückt muss der Abstand von Kandidatenlinie
        // zu Mittelpunkt kleiner sein als von Kandidatenlinie zu erster Linie.
    }

    // TODO: Use groups-pair with smallest distance between their points.
    // TODO: Throw out picking groups by size --> Only gives information about well-detected lines, not whether they are part of the board.

    // TODO: Jeweils beide Angle Gruppen wieder in Zwei teilen, je nach Seite des Bretts.
    // Aus diesen zwei Gruppen jeweils zwei Elemente aus unterschiedlichen Gruppen als Paar nehmen und das Paar mit der kleinsten Distanz suchen --> Dieses ist das Paar, welches
    // die Felder umrandet.

    // TODO: Linien ausweiten, sodass sie sich an den Ecken treffen (optional, wenn ich das bounding rect verwende)
    // TODO: Rechteck bilden (bounding rect der vier Ausgangslinien)
    // TODO: Bildbereich auf Brett beschränken!

    // TODO: Bild rotieren, sodass die Ausrichtung stimmt
    // https://learnopencv.com/image-rotation-and-translation-using-opencv/
    // Point2f boardCenter((image.cols - 1) / 2.0, (image.rows - 1) / 2.0);
    // getRotationMatrix2D(boardCenter, angle, scale) - Center kann aus den Brettbegrenzungen errechnet werden oder wie oben
    // warpAffine(image, rotated_image, rotation_matix, image.size());
    imshow("Lines", linesImg);
    imshow(w_name, dst);

    std::cout << std::endl;

    //--> 4 Linien erkennen, welche Brett bilden --> boundingRect!! --> Testen (imshow) --> Durch Größe der kleineren Felder (Linien) erkennen - Brett-Boundingrect muss
    //    ca. 8,xxx mal so groß sein wie Feldseite

    // TODO: Works in Debug but not Release --> Detects way more lines (also inner lines) in release --> Could be prevented by first checking for squares
    // https://stackoverflow.com/questions/36797737/opencv-result-changes-between-debug-release-and-on-other-machine

    return Board();
}

void process_images(std::vector<cv::Mat>& imgs) {

    for (cv::Mat img : imgs) {
        std::cout << "Processing..." << std::endl;
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
    //cap.open("C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221213_20_27_35_Pro.mp4");
    //cap.open("C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221213_20_28_03_Pro.mp4");
    //cap.open("C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221214_21_16_23_Pro.mp4");
    //cap.open("C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221214_21_16_49_Pro.mp4"); // TODO: Remove
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
        cap.open("C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221213_20_28_03_Pro.mp4");
        //cap.open("C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221213_20_27_35_Pro.mp4");
    }
    cap >> frame;
    cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    threshold(grayFrame, binFrame, 40, 40, THRESH_BINARY);
    try {
        if (countNonZero(binFrame) < 100 || frame.empty() || !opened) {
            std::cerr << "ERROR! Unable to open camera. Using video recording.\n";
            cap.release();
            cap.open("C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221213_20_28_03_Pro.mp4");
        }
        else {
            std::cout << "Nonzero: " << countNonZero(binFrame) << std::endl;
        }
    }
    catch (std::exception& e) {
        std::cerr << "ERROR! Unable to open camera. Using video recording.\n";
        cap.release();
        cap.open("C:\\Users\\sever\\OneDrive - bwedu\\6. Semester\\CV\\Labor\\Aufnahmen\\WIN_20221213_20_28_03_Pro.mp4");
    }
    return cap;
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

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    cv::Mat frame;
    cv::VideoCapture cap = select_camera();

    // TODO: Möglicher Ersatz für resize calls:
    //cap.set(3, defaultWidth);
    //cap.set(4, defaultHeight);
    //cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
    // TODO: Test

    float fps = 20.f;
    float frame_time = 1000.f / fps;
    int amount_frames = 4;
    std::vector<cv::Mat> frames;
    frames.reserve(amount_frames);

    cv::namedWindow(w_name);

    cv::setMouseCallback(w_name, onClickLive, 0);

    cv::namedWindow("Lines");

    cv::setMouseCallback("Lines", onClickLines, 0);

    cv::namedWindow("Rotated");
    cv::setMouseCallback("Rotated", onClickLines, 0);

    // TODO: cv::findChessboardCorners() https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    //cv::namedWindow("ChessCorners");
    //while (frame.empty()) {
    //    cap >> frame;
    //}
    //bool found = cv::findChessboardCorners(frame, cv::Size(8,8), frame);
    //if (found) {
    //    imshow("ChessCorners", frame);
    //}
    //std::cout << "Found: " << found << std::endl;

    // 1. Detect board
    // 2. Detect squares
    // 3. Determine orientation
    // 4. Place pieces
    // 5. Follow pieces
    
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Waiting for the webcam to focus

    //Board b = getBoard(cap); // Replace later with loop below
    // TODO: Keep trying to locate the board
    Board b = Board();
    while (!b.located) {
        b = getBoard(cap);
        cv::waitKey(100);
    }
    //while (true) {
    //    b = getBoard(cap);
    //    cv::waitKey(100);
    //}

    // TODO: Am Anfang, wenn Brett erkannt wird ein Bild speichern, das zum Vergleich benutzt wird während des Spiels - Pixelunterschiede berechnen, um zu erkennen ob eine
    // Figur auf Feld steht. (Aber zuerst rotieren, sodass es gerade ist, damit man die Rechtecke leichter einteilen kann! Feldbreite sollte schon von Bretterkennung bekannt sein.

    while (cv::waitKey(200)) {
        std::cout << "";
    }

    //while (cap.isOpened()) {
    //    for (int i = 0; i < amount_frames; i++) {
    //        cap >> frame;
    //        frames.push_back(frame);
    //        if (frame.empty()) {
    //            cerr << "ERROR! Blank frame grabbed\n";
    //            break;
    //        }
    //        //imshow("Live", frame);

    //        cv::waitKey(frame_time);
    //        //frame.release();
    //    }

    //    process_images(frames);

    //    frames.clear();

    //    // TODO: Frames timeout

    //}

    cap.release();

    std::cout << "Waiting for keypress..." << std::endl;
    cv::waitKey(0);

    return 0;
    
}
