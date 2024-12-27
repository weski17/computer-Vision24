#include "Person.hpp"
#include <opencv2/opencv.hpp>

/**
 * @class Person
 * @brief Repräsentiert eine verfolgte Person mit ID, Kontur und Schwerpunkt.
 */

// Standardkonstruktor
Person::Person() : id(-1), centroid(cv::Point(0, 0)) {}

// Konstruktor mit ID und Kontur
Person::Person(int id, const std::vector<cv::Point>& contour)
    : id(id), contour(contour) {
    if (!contour.empty()) {
        // Schwerpunkt basierend auf Kontur berechnen
        cv::Moments m = cv::moments(contour);
        centroid = cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
    } else {
        centroid = cv::Point(0, 0);
    }
}

// Destruktor
Person::~Person() {}

// Getter für die ID
int Person::getId() const {
    return id;
}

// Setter für die ID
void Person::setId(int id) {
    this->id = id;
}

// Getter für die Kontur
const std::vector<cv::Point>& Person::getContour() const {
    return contour;
}

// Setter für die Kontur und automatische Berechnung des Schwerpunkts
void Person::setContour(const std::vector<cv::Point>& contour) {
    this->contour = contour;
    if (!contour.empty()) {
        cv::Moments m = cv::moments(contour);
        centroid = cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
    } else {
        centroid = cv::Point(0, 0);
    }
}

// Getter für den Schwerpunkt
cv::Point Person::getCentroid() const {
    return centroid;
}

// Setter für den Schwerpunkt
void Person::setCentroid(const cv::Point& centroid) {
    this->centroid = centroid;
}
