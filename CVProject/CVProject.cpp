//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/core/utils/logger.hpp>
//#include <opencv2/calib3d.hpp>
//#include <algorithm>
#include "CVProject.h"


void getSquareData(VideoCapture& cap, Mat& refImg, Game& g, double squareLen, bool& properlyRotated) {
    int rowPx, colPx;
    int nCols = 8;
    int nRows = 8;
    int nSquares = 64;
    Mat frame;
    cap >> frame; // Now containing pieces
    Mat squareImg;
    char fileN, rankN;
    Scalar green = Scalar(255, 0, 0);
    Mat rotNoBorder = refImg.clone() - green;
    imshow("No green", rotNoBorder);
    int colorSum;
    int minMean = INT_MAX;
    int maxMean = -1;
    int whiteSquareThresh = 150;
    bool isWhite = false;
    if (g.squares.size() > 1) { g.squares.clear();}
    //vector<Scalar> meanColors;
    for (int row = 0; row < nRows; row++) {
        for (int col = 0; col < nCols; col++) {
            fileN = fileMap[col];
            rankN = (char)(row + 1 + '0');
            rowPx = (nRows - 1 - row) * squareLen;
            colPx = col * squareLen;
            Rect roi = Rect(Point(colPx, rowPx), Point(colPx + squareLen, rowPx + squareLen));
            squareImg = rotNoBorder(roi);
            cout << fileN << rankN << " --> " << rowPx << ", " << colPx;
            Scalar meanCol = mean(squareImg);
            colorSum = meanCol[0] + meanCol[1] + meanCol[2];
            cout << "\t" << meanCol << ",\tSum=" << colorSum << endl;
            if (colorSum < minMean) { minMean = colorSum; }
            if (colorSum > maxMean) { maxMean = colorSum; }
            if (colorSum > whiteSquareThresh) {
                isWhite = true;
            }
            else { isWhite = false; }
            if (row == 0 && col == 0 && !isWhite) {
                properlyRotated = true;
                // cout << "Was properly rotated" << endl;
            }
            else properlyRotated = false;
            g.squares.push_back(Square(Point(colPx, rowPx), roi, fileN, rankN, meanCol, squareImg, isWhite));
            //if (row == 1 && col == 0) circle(rotated_roi, Point(colPx, rowPx), 1, cv::Scalar(0, 255, 0), 2, 8, 0);
        }
    }
    if (g.squares[0].isWhite) {
        properlyRotated = false;
    }
    else { properlyRotated = true; }
}


Board getBoard(cv::VideoCapture& cap, Mat& refImg, Game& game) {
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

        double angle1 = avgAngles[p.first];
        std::vector<cv::Point> rotatedCorners(corners); // Create copy of corners
        cv::Mat rotated = rotate_image(frame, angle1, rotatedCorners);
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

        int squareLen = roi.width / 8; // Ist nicht schlimm wenn ein paar Pixel übrig bleiben

        Board b = Board(corners, roi);
        game = Game(b);
        bool properlyRotated = false;
        getSquareData(cap, refImg, game, squareLen, properlyRotated);
        if (!properlyRotated) {
            refImg = rotate_image(refImg, CV_PI/2., rotatedCorners);
            getSquareData(cap, refImg, game, squareLen, properlyRotated);
        }

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


bool detectPieces(VideoCapture& cap, Game& g, Mat& rotated_roi) {
    cout << "\nDetecting pieces..." << endl;
    Board b = g.board;

    for (Square s : g.squares) {
        cout << s.name << " ";
    }

    //imshow("Rotated roi", rotated_roi);

    return true;
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

    Mat referenceImg; // Achtung: Könnte Figuren enthalten!

    Board b = Board();
    Game g = Game();

    bool ending = false;
    bool piecesAccepted = false;
    while(!ending) {
        while (!b.located) {
            b = getBoard(cap, referenceImg, g);
            cv::waitKey(200);
        
            if (b.located) {
                cout << "\nBoard detection finished. If the board was correctly located, please place pieces on the board and press [Enter]." << endl;
                cout << "Accept board? [Enter] - Any other key to discard and try again..." << endl;
                char key = (char)waitKey(0); // Muss auf einem der Namedwindows sein, nicht auf der Konsole!
                //cout << "Key: " << (int)key << endl;
                if ((char)27 == key) { // Exit on Esc-Button
                    cout << "\nExiting..." << endl;
                    return 0;
                }
                if ((char)13 == key) { // Enter key
                    cout << "Detected board accepted." << endl;
                    break;
                }
                else {
                    cout << "Board detection discarded. Trying again..." << endl;
                    b.located = false;
                }
            }
        }

        while (!piecesAccepted) {
            piecesAccepted = detectPieces(cap, g, referenceImg);
            cv::waitKey(200);
            
            if (piecesAccepted) {
                cout << "\nAccept piece analysis? [Enter] - Any other key to discard and try again..." << endl;
                char key = waitKey(0);
                if ((char)27 == key) { // Esc-Button
                    cout << "\nExiting..." << endl;
                    return 0;
                }
                if ((char)13 == key) { // Enter key
                    cout << "Piece detection accepted." << endl;
                    break;
                }
                else {
                    cout << "Piece detection discarded. Trying again..." << endl;
                    piecesAccepted = false;
                }
            }
        }
    }

    // TODO: Detect pieces
    // While loop with possibility to do all over

    while (cv::waitKey(200)) {
        std::cout << ""; // TODO: Remove --> Put to end to keep clicking coordinates
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
