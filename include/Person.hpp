#ifndef PERSON_HPP
#define PERSON_HPP

#include <vector>
#include <opencv2/opencv.hpp>

class Person {
private:
    int id; // Einzigartige ID des Tracks
    std::vector<cv::Point> contour; // Kontur des Objekts
    cv::Point centroid; // Schwerpunkt des Objekts
    cv::Rect boundingBox; // Bounding Box der Kontur
    double area; // Fläche der Kontur
    double aspectRatio; // Breite/Höhe der Bounding Box
    double compactness; // Kompaktheit (Formparameter)
    std::vector<cv::KeyPoint> keypoints; // Keypoints der Kontur
    cv::Mat hsvHistogram; // HSV-Histogramm für Farbmerkmale
    std::vector<cv::Point2f> trackedPoints; // Getrackte Punkte (Optical Flow)

public:
    // Standardkonstruktor
    Person();

    // Konstruktor mit ID und Kontur
    Person(int id, const std::vector<cv::Point>& contour);

    // Destruktor
    ~Person();

    // Getter und Setter
    int getId() const;
    void setId(int id);

    const std::vector<cv::Point>& getContour() const;
    void setContour(const std::vector<cv::Point>& contour, const cv::Mat& frame);

    cv::Point getCentroid() const;
    void setCentroid(const cv::Point& centroid);

    cv::Rect getBoundingBox() const;
    double getArea() const;
    double getAspectRatio() const;
    double getCompactness() const;

    const std::vector<cv::KeyPoint>& getKeypoints() const;
    void setKeypoints(const std::vector<cv::KeyPoint>& keypoints);

    const std::vector<cv::Point2f>& getTrackedPoints() const;
    void setTrackedPoints(const std::vector<cv::Point2f>& points);

    cv::Mat getHsvHistogram() const;
    void setHsvHistogram(const cv::Mat& histogram);

    // Methoden zur Merkmalsextraktion
    void calculateProperties();
    void extractKeypoints(const std::vector<cv::Point>& contour);
    void computeHsvHistogram(const cv::Mat& frame);
};

#endif // PERSON_HPP
