/**
 * @file Person.cpp
 * @brief Implementierung der Person-Klasse für das Tracking einzelner Personen.
 *
 * Diese Datei enthält die Implementierung der Person-Klasse, die Eigenschaften wie
 * Kontur, Bounding Box, Centroid, Keypoints und HSV-Histogramm verwaltet. Sie bietet
 * Methoden zur Berechnung von Eigenschaften, Extraktion von Keypoints und Berechnung
 * von HSV-Histogrammen.
 */

#include "Person.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

// Standardkonstruktor
/**
 * @brief Standardkonstruktor der Person-Klasse.
 *
 * Initialisiert eine Person mit Standardwerten. Die ID wird auf -1 gesetzt, und alle
 * numerischen Eigenschaften werden auf 0. Der Centroid wird auf den Punkt (0,0) gesetzt.
 */
Person::Person() 
    : id(-1), 
      centroid(cv::Point(0, 0)), 
      area(0.0), 
      aspectRatio(0.0), 
      compactness(0.0) {}

// Konstruktor mit ID und Kontur
/**
 * @brief Konstruktor der Person-Klasse mit ID und Kontur.
 *
 * Initialisiert eine Person mit einer eindeutigen ID und einer gegebenen Kontur.
 * Berechnet anschließend die abhängigen Eigenschaften wie Fläche, Bounding Box, Centroid,
 * Aspektverhältnis und Kompaktheit.
 *
 * @param id Eindeutige ID der Person.
 * @param contour Kontur der Person als Vektor von Punkten.
 */
Person::Person(int id, const std::vector<cv::Point>& contour) 
    : id(id), 
      contour(contour) {
    calculateProperties();
}

// Destruktor
/**
 * @brief Destruktor der Person-Klasse.
 *
 * Gibt alle Ressourcen frei, die von der Person-Klasse verwendet werden.
 */
Person::~Person() {}

// Getter und Setter

/**
 * @brief Gibt die ID der Person zurück.
 * @return Die eindeutige ID der Person.
 */
int Person::getId() const {
    return id;
}

/**
 * @brief Setzt die ID der Person.
 * @param id Die eindeutige ID, die der Person zugewiesen werden soll.
 */
void Person::setId(int id) {
    this->id = id;
}

/**
 * @brief Gibt die Kontur der Person zurück.
 * @return Eine Referenz auf den Vektor der Punkte, die die Kontur definieren.
 */
const std::vector<cv::Point>& Person::getContour() const {
    return contour;
}

/**
 * @brief Setzt die Kontur der Person und berechnet abhängige Eigenschaften.
 *
 * Aktualisiert die Kontur der Person und berechnet anschließend die
 * abhängigen Eigenschaften wie Fläche, Bounding Box, Centroid, Aspektverhältnis
 * und Kompaktheit. Zusätzlich wird das HSV-Histogramm basierend auf dem aktuellen Frame
 * berechnet.
 *
 * @param contour Neue Kontur der Person als Vektor von Punkten.
 * @param frame Aktuelles Videoframe, das für die Histogrammberechnung verwendet wird.
 */
void Person::setContour(const std::vector<cv::Point>& contour, const cv::Mat& frame) {
    this->contour = contour;
    calculateProperties();
    computeHsvHistogram(frame);
}

/**
 * @brief Gibt den Centroid (Schwerpunkt) der Person zurück.
 * @return Der Centroid der Person als cv::Point.
 */
cv::Point Person::getCentroid() const {
    return centroid;
}

/**
 * @brief Setzt den Centroid (Schwerpunkt) der Person.
 * @param centroid Der neue Centroid als cv::Point.
 */
void Person::setCentroid(const cv::Point& centroid) {
    this->centroid = centroid;
}

/**
 * @brief Gibt die Bounding Box der Kontur zurück.
 * @return Die Bounding Box als cv::Rect.
 */
cv::Rect Person::getBoundingBox() const {
    return boundingBox;
}

/**
 * @brief Gibt die Fläche der Kontur zurück.
 * @return Die Fläche der Kontur als double.
 */
double Person::getArea() const {
    return area;
}

/**
 * @brief Gibt das Aspektverhältnis (Breite/Höhe) der Bounding Box zurück.
 * @return Das Aspektverhältnis als double.
 */
double Person::getAspectRatio() const {
    return aspectRatio;
}

/**
 * @brief Gibt die Kompaktheit der Kontur zurück.
 * @return Die Kompaktheit als double.
 */
double Person::getCompactness() const {
    return compactness;
}

/**
 * @brief Gibt die Keypoints der Kontur zurück.
 * @return Eine Referenz auf den Vektor der Keypoints.
 */
const std::vector<cv::KeyPoint>& Person::getKeypoints() const {
    return keypoints;
}

/**
 * @brief Setzt die Keypoints der Kontur.
 * @param keypoints Der Vektor der Keypoints, die der Kontur zugeordnet werden sollen.
 */
void Person::setKeypoints(const std::vector<cv::KeyPoint>& keypoints) {
    this->keypoints = keypoints;
}

/**
 * @brief Gibt die getrackten Punkte der Person zurück.
 * @return Eine Referenz auf den Vektor der getrackten Punkte als cv::Point2f.
 */
const std::vector<cv::Point2f>& Person::getTrackedPoints() const {
    return trackedPoints;
}

/**
 * @brief Setzt die getrackten Punkte der Person.
 * @param points Der Vektor der Punkte, die getrackt werden sollen.
 */
void Person::setTrackedPoints(const std::vector<cv::Point2f>& points) {
    trackedPoints = points;
}

/**
 * @brief Gibt das HSV-Histogramm der Person zurück.
 * @return Das HSV-Histogramm als cv::Mat.
 */
cv::Mat Person::getHsvHistogram() const {
    return hsvHistogram;
}

/**
 * @brief Setzt das HSV-Histogramm der Person.
 * @param histogram Das HSV-Histogramm, das der Person zugewiesen werden soll.
 */
void Person::setHsvHistogram(const cv::Mat& histogram) {
    hsvHistogram = histogram;
}

// Eigenschaften berechnen

/**
 * @brief Berechnet die Eigenschaften der Kontur (Fläche, Bounding Box, Centroid, Aspektverhältnis, Kompaktheit).
 *
 * Diese Methode berechnet die Fläche der Kontur, die Bounding Box, den Centroid (Schwerpunkt),
 * das Aspektverhältnis (Breite/Höhe) und die Kompaktheit (4π * Fläche / Umfang²) der Kontur.
 * Sie wird aufgerufen, wenn die Kontur gesetzt oder aktualisiert wird.
 */
void Person::calculateProperties() {
    if (contour.empty()) return;

    // Fläche berechnen
    area = cv::contourArea(contour);

    // Bounding Box berechnen
    boundingBox = cv::boundingRect(contour);

    // Schwerpunkt berechnen
    cv::Moments moments = cv::moments(contour);
    if (moments.m00 != 0) {
        centroid = cv::Point(static_cast<int>(moments.m10 / moments.m00),
                             static_cast<int>(moments.m01 / moments.m00));
    } else {
        centroid = cv::Point(0, 0);
    }

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

/**
 * @brief Extrahiert Keypoints basierend auf der Kontur.
 *
 * Diese Methode extrahiert Keypoints aus der Kontur der Person. Sie wählt Punkte in regelmäßigen
 * Abständen entlang der Kontur aus, um eine gleichmäßige Verteilung der Keypoints zu gewährleisten.
 *
 * @param contour Die Kontur der Person als Vektor von Punkten.
 */
void Person::extractKeypoints(const std::vector<cv::Point>& contour) {
    std::vector<cv::KeyPoint> contourKeypoints;
    int step = std::max(1, static_cast<int>(contour.size() / 100)); // Schrittgröße bestimmen
    for (size_t i = 0; i < contour.size(); i += step) {
        cv::Point2f pt = contour[i];
        cv::KeyPoint keyPoint(pt, 2.0f); // Erstelle KeyPoint mit einer Größe von 2.0
        contourKeypoints.push_back(keyPoint);
    }
    keypoints = contourKeypoints; // Speichere extrahierte Keypoints
}

/**
 * @brief Berechnet das HSV-Histogramm für die Kontur innerhalb eines Frames.
 *
 * Diese Methode extrahiert das Region of Interest (ROI) basierend auf der Bounding Box der Kontur,
 * konvertiert es in den HSV-Farbraum und berechnet das 2D-Histogramm für die Hue- und Saturation-Kanäle.
 * Anschließend wird das Histogramm normalisiert.
 *
 * @param frame Das aktuelle Videoframe, das für die Histogrammberechnung verwendet wird.
 */
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
