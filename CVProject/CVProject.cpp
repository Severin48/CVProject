//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/core/utils/logger.hpp>
//#include <opencv2/calib3d.hpp>
//#include <algorithm>
#include "CVProject.h"

using namespace std;

double meanWhite = -1;
double meanBlack = -1;


void getSquareData(VideoCapture& cap, Mat& refImg, Game& g, bool& properlyRotated) {
    cout << "Getting square data..." << endl;
    int rowPx, colPx;
    int nCols = 8;
    int nRows = 8;
    int nSquares = 64;
    Mat frame;
    cap >> frame;
    Mat squareImg, squareImgNoGreen;
    char fileN, rankN;
    Scalar green = Scalar(255, 0, 0);
    Mat rotNoBorder = refImg.clone(); // - green;
    int colorSum;
    int minMean = INT_MAX;
    int maxMean = -1;
    int whiteSquareThresh = 200;
    bool isWhite = false;
    if (g.squares.size() > 1) { g.squares.clear();}
    int counter = 0;
    int sqWidth = rotNoBorder.cols / 8;
    int sqHeight = rotNoBorder.rows / 8;
    for (int row = 0; row < nRows; row++) {
        for (int col = 0; col < nCols; col++) {
            fileN = fileMap[col];
            rankN = (char)(row + 1 + '0');
            rowPx = (nRows - 1 - row) * sqHeight;
            colPx = col * sqWidth;
            Rect roi = Rect(Point(colPx, rowPx), Point(colPx + sqWidth, rowPx + sqHeight));
            squareImg = rotNoBorder(roi);
            squareImgNoGreen = squareImg - green;
            cout << fileN << rankN << " --> " << rowPx << ", " << colPx;
            Scalar meanCol = mean(squareImgNoGreen);
            colorSum = meanCol[0] + meanCol[1] + meanCol[2];
            cout << "\t" << meanCol << ",\tSum=" << colorSum << endl;
            if (colorSum < minMean) { minMean = colorSum; }
            if (colorSum > maxMean) { maxMean = colorSum; }
            if (colorSum > whiteSquareThresh) {
                isWhite = true;
            }
            else { isWhite = false; }
            if (row == 3 && col == 4) {
                //rectangle(rotNoBorder, roi, cv::Scalar(0, 0, 255));
                if (isWhite) {
                    meanWhite = colorSum;
                    meanBlack = g.squares[g.squares.size() - 1].meanSum;
                }
                else {
                    meanBlack = colorSum;
                    meanWhite = g.squares[g.squares.size() - 1].meanSum;
                }
                cout << "MeanBlack=" << meanBlack << ", meanWhite=" << meanWhite << endl;
            }
            g.squares.push_back(Square(Point(colPx, rowPx), roi, fileN, rankN, meanCol, squareImg, isWhite, colorSum));
            //if (row == 1 && col == 0) circle(rotated_roi, Point(colPx, rowPx), 1, cv::Scalar(0, 255, 0), 2, 8, 0);
            if (!((row == 3 && col == 3) || (row == 3 && col == 4))) {
                if (counter % 2 == 0) { isWhite = false; }
                else { isWhite = true; }
            }
            if (row == 3 && col == 3) {
                if (isWhite) properlyRotated = false;
                else properlyRotated = true;
            }
            
            counter++;
            refImg = rotNoBorder;
            //imshow("ROI", refImg);
        }
    }
}


Board getBoard(cv::VideoCapture& cap, Mat& refImg, Game& game, vector<double>& rotations) {
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
        std::vector<cv::Vec4i> g1Lines;
        std::vector<cv::Vec4i> g2Lines;
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

        // imshow("Lines", linesImg);

        float distance_percentage = 0.85;

        bool added = false;
        cout << "G1 lines: " << g1Lines.size() << endl;
        std::pair<Vec4i, Vec4i> farthestLinesG1;
        double g1maxDist = getMaxDistLines(g1Lines, farthestLinesG1);
        double dist = 0;
        vector<Vec4i> filteredLines;
        vector<Vec4i> g1LinesFiltered;
        for (Vec4i outerLine : g1Lines) {
            added = false;
            if (outerLine == farthestLinesG1.first || outerLine == farthestLinesG1.second) { continue; }
            for (Vec4i innerLine : g1Lines) {
                if (innerLine == farthestLinesG1.first || innerLine == farthestLinesG1.second) { continue; }
                dist = distBetweenLines(outerLine, innerLine);
                if (dist >= distance_percentage * g1maxDist) {
                    if (!added) {
                        filteredLines.push_back(outerLine);
                        g1LinesFiltered.push_back(outerLine);
                        added = true;
                    }
                }
            }
        }
        Mat filteredLinesImg(frame);
        for (Vec4i l : filteredLines) {
            line(filteredLinesImg, Point(l[0], l[1]), Point(l[2], l[3]), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

        added = false;
        cout << "G2 lines: " << g2Lines.size() << endl;
        std::pair<Vec4i, Vec4i> farthestLinesG2;
        vector<Vec4i> g2LinesFiltered;
        double g2maxDist = getMaxDistLines(g2Lines, farthestLinesG2);
        filteredLines.clear();
        for (Vec4i outerLine : g2Lines) {
            added = false;
            if (outerLine == farthestLinesG2.first || outerLine == farthestLinesG2.second) { continue; }
            for (Vec4i innerLine : g2Lines) {
                if (innerLine == farthestLinesG2.first || innerLine == farthestLinesG2.second) { continue; }
                dist = distBetweenLines(outerLine, innerLine);
                if (dist >= distance_percentage * g2maxDist) {
                    if (!added) {
                        added = true;
                        filteredLines.push_back(outerLine);
                        g2LinesFiltered.push_back(outerLine);
                    }
                }
            }
        }
        for (Vec4i l : filteredLines) {
            line(filteredLinesImg, Point(l[0], l[1]), Point(l[2], l[3]), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

        //imshow("Filtered Lines", filteredLinesImg);
        cout << "Filtered lines: " << filteredLines.size() << endl;

        // TODO: Use filtered lines + filteredG1Lines + filteredG2Lines

        for (auto l : g1LinesFiltered) {
            cv::Point p1 = cv::Point(l[0], l[1]);
            cv::Point p2 = cv::Point(l[2], l[3]);
            distancesToCenterG1.push_back(euclideanDist(midpoint(p1, p2), boardCenter));
            //std::cout << euclideanDist(midpoint(p1, p2), boardCenter) << std::endl;
        }

        for (auto l : g2LinesFiltered) {
            cv::Point p1 = cv::Point(l[0], l[1]);
            cv::Point p2 = cv::Point(l[2], l[3]);
            distancesToCenterG2.push_back(euclideanDist(midpoint(p1, p2), boardCenter));
        }

        if (distancesToCenterG1.size() < 2 || distancesToCenterG2.size() < 2) return Board(); // Not enough board limiting lines found.

        int smallestIdx = std::min_element(distancesToCenterG1.begin(), distancesToCenterG1.end()) - distancesToCenterG1.begin();
        double smallest = distancesToCenterG1[smallestIdx];
        double second = DBL_MAX;
        int secondIdx = -1;
        cv::Point midSmallest = midpoint(g1LinesFiltered[smallestIdx]);
        for (int i = 0; i < distancesToCenterG1.size(); i++) {
            double distToFirstLine = euclideanDist(midSmallest, midpoint(g1LinesFiltered[i]));
            double d = distancesToCenterG1[i];
            if (d <= second && i != smallestIdx && d < distToFirstLine) {
                secondIdx = i;
                second = d;
            }
        }
        
        std::cout << smallestIdx << ": " << smallest << ", " << secondIdx << ": " << second << std::endl;

        if (smallestIdx == -1 || secondIdx == -1) return Board();

        boardLimits.push_back(g1LinesFiltered[smallestIdx]);
        boardLimits.push_back(g1LinesFiltered[secondIdx]);

        smallestIdx = std::min_element(distancesToCenterG2.begin(), distancesToCenterG2.end()) - distancesToCenterG2.begin();
        smallest = distancesToCenterG2[smallestIdx];
        secondIdx = -1;
        second = DBL_MAX;
        midSmallest = midpoint(g2LinesFiltered[smallestIdx]);
        for (int i = 0; i < distancesToCenterG2.size(); i++) {
            double distToFirstLine = euclideanDist(midSmallest, midpoint(g2LinesFiltered[i]));
            double d = distancesToCenterG2[i];
            if (d <= second && i != smallestIdx && d < distToFirstLine) {
                secondIdx = i;
                second = d;
            }
        }

        std::cout << smallestIdx << ": " << smallest << ", " << secondIdx << ": " << second << std::endl;

        if (smallestIdx == -1 || secondIdx == -1) return Board();

        boardLimits.push_back(g2LinesFiltered[smallestIdx]);
        boardLimits.push_back(g2LinesFiltered[secondIdx]);

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

        rotations.push_back(avgAngles[p.first]);
        std::vector<cv::Point> rotatedCorners(corners); // Create copy of corners
        cv::Mat rotated = rotate_image(frame, rotations[0], rotatedCorners);
        cout << "Rotated corners:\n" << rotatedCorners << endl;
        cv::Point center = cv::Point(frame.cols / 2, frame.rows / 2);

        int tolerancePxs = 20;
        Point topLeft = Point(INT_MAX - tolerancePxs, INT_MAX - tolerancePxs);
        Point botRight = Point(0, 0);
        int minSum = INT_MAX;
        int maxSum = -1;
        for (Point p : rotatedCorners) {
            int sum = p.x + p.y;
            if (sum < minSum) { minSum = sum; topLeft = p; }
            if (sum > maxSum) { maxSum = sum; botRight = p; }
        }

        cout << "Top left: " << topLeft << ", botRight: " << botRight << endl;

        cv::Rect roi = cv::Rect(topLeft, botRight);
        if (roi.width < 20 || roi.height < 20) { cerr << "ROI too small" << endl; return Board(); }

        refImg = rotated(roi);
        //imshow("Rotated roi here", refImg);
        Scalar green = cv::Scalar(255, 0, 0);
        line(dst, corners[0], corners[1], green, 1, cv::LINE_AA);
        line(dst, corners[0], corners[2], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        line(dst, corners[3], corners[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        line(dst, corners[3], corners[2], cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

        int longestSidePx = min(roi.width, roi.height);
        int squareLen = roi.width / 8; // Ist nicht schlimm wenn ein paar Pixel übrig bleiben

        Board b = Board(corners, roi);
        game = Game(b);
        game.roi = roi;
        bool properlyRotated = false;
        getSquareData(cap, refImg, game, properlyRotated);
        if (!properlyRotated) {
            //correctionRotationNeeded = true;
            rotations.push_back(CV_PI / 2.);
            refImg = rotate_image(refImg, CV_PI/2., rotatedCorners);
            getSquareData(cap, refImg, game, properlyRotated);
        }
        //else { correctionRotationNeeded = false; }

        //line(rotated, rotatedCorners[0], rotatedCorners[1], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        //line(rotated, rotatedCorners[0], rotatedCorners[2], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        //line(rotated, rotatedCorners[3], rotatedCorners[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        //line(rotated, rotatedCorners[3], rotatedCorners[2], cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

        //imshow("Rotated", rotated);
        imshow(w_name, dst);
        return Board(corners, roi);
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
    //imshow("Lines", linesImg);
    imshow(w_name, dst);

    std::cout << std::endl;

    //--> 4 Linien erkennen, welche Brett bilden --> boundingRect!! --> Testen (imshow) --> Durch Größe der kleineren Felder (Linien) erkennen - Brett-Boundingrect muss
    //    ca. 8,xxx mal so groß sein wie Feldseite

    // TODO: Works in Debug but not Release --> Detects way more lines (also inner lines) in release --> Could be prevented by first checking for squares
    // https://stackoverflow.com/questions/36797737/opencv-result-changes-between-debug-release-and-on-other-machine

    return Board();
}


bool detectPieces(VideoCapture& cap, Game& g, bool& needsFlip, Mat& roiImg) {
    cout << "\nDetecting pieces..." << endl;
    Board b = g.board;

    int squareLen = b.roi.width/8.;
    int rowPx, colPx;
    int nCols = 8;
    int nRows = 8;
    int nSquares = 64;
    int colorSum;

    Mat squareImg;
    Scalar green = Scalar(255, 0, 0);
    Mat rotNoBorder = roiImg.clone() - green;
    //imshow("Detecting Pieces ROI", rotNoBorder);

    int oppositeColorPieceThresh = 60;
    for (Square& s : g.squares) {
        squareImg = rotNoBorder(s.rect);
        Scalar meanCol = mean(squareImg);
        colorSum = meanCol[0] + meanCol[1] + meanCol[2];
        double diffBlack = abs(colorSum - meanBlack);
        double diffWhite = abs(colorSum - meanWhite);
        if (s.name == "a1") {
            cout << "Diffblack: " << diffBlack << endl;
            if (diffBlack >= oppositeColorPieceThresh) {
                cout << "Bottom left square is occupied by a white piece." << endl;
                needsFlip = false;
            }
            else needsFlip = true;
        }
    }
    //roiImg = rotNoBorder;

    return true;
}

bool isPawn(Piece p) {
    if (p.name[0] > '0' && p.name[0] < '9') return true; // TODO: Test
    else return false;
}

void assignPieces(Game& g, Mat& roiImgSrc, bool needsFlip, vector<double>& rotations) {
    cout << "\n\nAssigning pieces to squares..." << endl;
    //imshow("Before flip", roiImgSrc);
    if (needsFlip) {
        rotations.push_back(CV_PI);
        roiImgSrc = rotate_image(roiImgSrc, CV_PI);
    }

    //imshow("After flip", roiImgSrc);
    Mat roiImg = roiImgSrc.clone();
    if (g.pieces.size() > 1) g.pieces.clear();
    for (Square& s : g.squares) {
        if (s.rank == '1') {
            string pieceName = filePieceMap[s.file];
            g.pieces.push_back(Piece(pieceName, true));
            s.occupiedByWhite = true;
            s.piece = pieceName[0];
            s.occupied = true;
        }
        if (s.rank == '2') {
            s.occupiedByWhite = true;
            s.piece = s.file;
            s.occupied = true;
            g.pieces.push_back(Piece(s.name, true));
        }
        if (s.rank == '7') {
            s.occupiedByWhite = false;
            s.piece = s.file;
            s.occupied = true;
            g.pieces.push_back(Piece(s.name, false));
        }
        if (s.rank == '8') {
            string pieceName = filePieceMap[s.file];
            g.pieces.push_back(Piece(pieceName, false));
            s.occupiedByWhite = false;
            s.piece = pieceName[0];
            s.occupied = true;
        }
        s.refreshImg(roiImg);
        if (s.piece != NULL) {
            cout << s.name << ": " << s.piece << " ";
            //s.showRect(roiImg);
        }
    }
    imshow("RoiImg", roiImg);
}

void trackPieces(VideoCapture& cap, Game& g, Mat& refImg, bool prevPawnTwoAdvances, char file, vector<double> rotations) {
    // Pieces markieren mit blauem Rechteck um Squares, welche eine Figur enthalten
    // + in einer anderen Funktion dann Figuren tracken --> Schleife, in der immer analysiert wird, wo Figuren stehen --> Wenn in einem Frame mehrere fehlen verwerfe Frame
    // --> Ist wahrscheinlich von Hand verdeckt - es darf nur eine Figur verstellt sein - Bilder von allen anderen machen und schauen ob sie noch ungefähr gleich sind!
    // Einfache Alternative: Jeden Zug mit Enter bestätigen - aus Zeitgründen lieber das machen
    // Art der Figur merken! Und printen z.B. Knight moved from B1 to C3
    // Optional: PGN mit Python verbinden

    Mat trackImg = refImg.clone();

    Mat frame;
    while (frame.empty()) {
        cap >> frame;
    }
    for (int i = 0; i < rotations.size(); i++) {
        frame = rotate_image(frame, rotations[i]);
        if (i == 0) {
            frame = frame(g.roi);
        }
    }
    Scalar green = (255, 0, 0);
    frame = frame - green;
    imshow("Frame", frame);
    //Mat currentImg = frame(g.roi);
    //imshow("Current", frame);
    Mat currentSquareImg, diffImg;
    Mat currentCopy;
    int meanSum, diff;
    double blackThresh = 0.07;
    double whiteThresh = 0.3;
    double totalDiffFactor;
    //cout << "\n\nDiffs: " << endl;

    // TODO: Zuerst hier noch nichts ändern erst wenn bestätigt ist, dass es nicht von Hand bedeckt ist
    double maxDiff = 0;
    int changed = 0;
    double gDiff, bDiff, rDiff, totalDiff;
    string maxName;
    for (Square& s : g.squares) {
        currentSquareImg = frame(s.rect);
        currentCopy = currentSquareImg.clone();
        absdiff(s.img, currentCopy, diffImg);
        threshold(diffImg, diffImg, 80, 255, cv::THRESH_BINARY);
        erode(diffImg, diffImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        Scalar meanCol = mean(diffImg);
        meanSum = meanCol[0] + meanCol[1] + meanCol[2];

        //for (int i = 0; i < s.img.rows; i++) {
        //    for (int j = 0; j < s.img.cols; j++) {
        //        std::cout << s.img.at<cv::Vec3b>(i, j)[0] << " " << s.img.at<cv::Vec3b>(i, j)[1] << " " << s.img.at<cv::Vec3b>(i, j)[2] << std::endl;
        //    }
        //}

        // TODO: Im Verhältnis setzen zu vorherigem Mean!!!!
        
        //gDiff = abs(meanCol[0] - s.meanCol[0]);
        //bDiff = abs(meanCol[1] - s.meanCol[1]);
        //rDiff = abs(meanCol[2] - s.meanCol[2]);
        //totalDiff = gDiff + bDiff + rDiff;
        totalDiffFactor = meanSum / (double)s.meanSum;

        cout << s.name << " --> Total DiffFactor=" << totalDiffFactor << endl;
        if (totalDiffFactor > maxDiff) {
            maxDiff = totalDiffFactor;
            maxName = s.name;
        }
        //if (meanSum > maxDiff) {
        //    maxDiff = meanSum;
        //    maxName = s.name;
        //}
        if (s.name == "d3") {
            imshow("Current"+s.name, currentSquareImg);
            imshow("Diff"+s.name, diffImg);
            cout << "Difference of " << s.name << "=" << totalDiffFactor << endl;
            imshow(s.name, s.img);
        }
        if (s.name == "a8") {
            imshow(s.name, diffImg);
            cout << "Here";
        }
        if (!s.isWhite && !s.occupiedByWhite) {
            if (totalDiffFactor > blackThresh) {
                cout << "Black on black changed: " << s.name << " --> " << totalDiffFactor << endl;
                changed++;
            }
        }
        else {
            if (totalDiffFactor > whiteThresh) {
                cout << "Changed: " << s.name << " --> " << totalDiffFactor << endl;
                changed++;
            }
        }

    }
    cout << "\nMax Diff: " << maxDiff << ", maxName=" << maxName << endl;
    cout << "Changed: " << changed << endl;
    if (changed > 3) {
        cout << "View obstructed... trying again" << endl;
        return;
    }

    // Hier bewegte Figuren bestimmen
    // TODO: Refresh square images
    // TODO: Am Ende das Referenzbild updaten
    // TODO: Update which squares are occupied by which piece and which color
    // Update vacated square (2 squares for en-passant) use parameters
    // TODO: Support castling

    for (Square& s : g.squares) {
        //cout << s.name << " occupied: " << s.occupied << " ";
        if (s.occupied) {
            s.showRect(trackImg);
        }
    }
    imshow("Tracking", trackImg);

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

    //cv::namedWindow("Lines");

    //cv::setMouseCallback("Lines", onClickLines, 0);

    //cv::namedWindow("Rotated");
    //cv::setMouseCallback("Rotated", onClickLines, 0);

    // 1. Detect board
    // 2. Detect squares
    // 3. Determine orientation
    // 4. Place pieces
    // 5. Follow pieces
    
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Waiting for the webcam to focus

    Mat referenceImg; // Achtung: Könnte Figuren enthalten!

    Board b = Board();
    Game g = Game();
    vector<double> rotations;
    bool correctionNeeded = false;
    Mat refImg;

    bool ending = false;
    bool piecesAccepted = false;
    while(!ending) {
        while (!piecesAccepted) {
            b = getBoard(cap, referenceImg, g, rotations);
            bool needsFlip = false;
            bool piecesValid = false;
            piecesValid = detectPieces(cap, g, needsFlip, referenceImg);
            //imshow("Here", roiImg);
            assignPieces(g, referenceImg, needsFlip, rotations);
            
            if (piecesValid) {
                //cout << "\nAccept piece analysis? [Enter] - Any other key to discard and try again..." << endl;
                cout << "\nRestart analysis? [r] - Any other key to keep tracking..." << endl;
                char key = waitKey(0);
                if ((char)27 == key) { // Esc-Button
                    cout << "\nExiting..." << endl;
                    return 0;
                }
                if ((char)114 == key) {
                    cout << "\nResetting board and restarting analysis..." << endl;
                    b = Board();
                    g = Game();
                    continue;
                }
                else {
                    bool prevPawnTwoAdvances = false;
                    char file = 'x';
                    while ((char)27 != waitKey(400)) {
                        trackPieces(cap, g, referenceImg, prevPawnTwoAdvances, file, rotations);
                    }
                    //trackPieces(g, referenceImg);
                    return 0;
                }
                //if ((char)13 == key) { // Enter key
                //    cout << "Piece detection accepted." << endl;
                //    break;
                //}
                // TODO: If pressed r: break and set boardAccepted = false + piecesValid = false, press r to restart the process
                //else {
                //    cout << "Piece detection discarded. Trying again..." << endl;
                //    piecesAccepted = false;
                //}
            }
        }
    }
    
    cap.release();

    //while (cv::waitKey(200)) {
    //    std::cout << "";
    //}

    std::cout << "Waiting for keypress..." << std::endl;
    cv::waitKey(0);

    return 0;
    
}
