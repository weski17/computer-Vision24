#include "Person.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

// Standardkonstruktor
Person::Person() : id(-1), centroid(cv::Point(0, 0)), area(0.0), aspectRatio(0.0), compactness(0.0) {}

// Konstruktor mit ID und Kontur
Person::Person(int id, const std::vector<cv::Point>& contour) : id(id), contour(contour) {
    calculateProperties();
}

// Destruktor
Person::~Person() {}

// Getter und Setter
int Person::getId() const {
    return id;
}

void Person::setId(int id) {
    this->id = id;
}

const std::vector<cv::Point>& Person::getContour() const {
    return contour;
}

void Person::setContour(const std::vector<cv::Point>& contour, const cv::Mat& frame) {
    this->contour = contour;
    calculateProperties();
    computeHsvHistogram(frame);
}

cv::Point Person::getCentroid() const {
    return centroid;
}

void Person::setCentroid(const cv::Point& centroid) {
    this->centroid = centroid;
}

cv::Rect Person::getBoundingBox() const {
    return boundingBox;
}

double Person::getArea() const {
    return area;
}

double Person::getAspectRatio() const {
    return aspectRatio;
}

double Person::getCompactness() const {
    return compactness;
}

const std::vector<cv::KeyPoint>& Person::getKeypoints() const {
    return keypoints;
}

void Person::setKeypoints(const std::vector<cv::KeyPoint>& keypoints) {
    this->keypoints = keypoints;
}

const std::vector<cv::Point2f>& Person::getTrackedPoints() const {
    return trackedPoints;
}

void Person::setTrackedPoints(const std::vector<cv::Point2f>& points) {
    trackedPoints = points;
}

cv::Mat Person::getHsvHistogram() const {
    return hsvHistogram;
}

void Person::setHsvHistogram(const cv::Mat& histogram) {
    hsvHistogram = histogram;
}

// Eigenschaften berechnen
void Person::calculateProperties() {
    if (contour.empty()) return;

    // Fläche berechnen
    area = cv::contourArea(contour);

    // Bounding Box berechnen
    boundingBox = cv::boundingRect(contour);

    // Aspektverhältnis (Breite/Höhe)
    if (boundingBox.height != 0) {
        aspectRatio = static_cast<double>(boundingBox.width) / boundingBox.height;
    } else {
        aspectRatio = 0.0;
    }

    // Kompaktheit (4π * Fläche / Umfang²)
    double perimeter = cv::arcLength(contour, true);
    if (perimeter != 0) {
        compactness = (4 * CV_PI * area) / (perimeter * perimeter);
    } else {
        compactness = 0.0;
    }
}

// Keypoints extrahieren
void Person::extractKeypoints(const std::vector<cv::Point>& contour) {
    cv::Rect boundingBox = cv::boundingRect(contour);
    cv::Mat mask = cv::Mat::zeros(boundingBox.size(), CV_8UC1);

    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, -boundingBox.tl());

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(mask, corners, 100, 0.01, 10);

    std::vector<cv::KeyPoint> cornerKeypoints;
    for (const auto& corner : corners) {
        cornerKeypoints.emplace_back(corner, 3.0f);
    }

    keypoints = cornerKeypoints;
}


void Person::computeHsvHistogram(const cv::Mat& frame) {
    if (boundingBox.area() == 0 || frame.empty()) return;

    cv::Mat roi = frame(boundingBox);
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    int hBins = 50, sBins = 60;
    int histSize[] = {hBins, sBins};

    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    const float* ranges[] = {hRanges, sRanges};

    int channels[] = {0, 1};

    cv::calcHist(&hsv, 1, channels, cv::Mat(), hsvHistogram, 2, histSize, ranges, true, false);
    cv::normalize(hsvHistogram, hsvHistogram, 0, 255, cv::NORM_MINMAX);
}
