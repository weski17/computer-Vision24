/**
 * @file Person.hpp
 * @brief Header-Datei für die Person-Klasse, die Eigenschaften und Methoden für das Tracking einzelner Personen enthält.
 *
 * Diese Klasse speichert Informationen über die Kontur, Bounding Box, Keypoints, und andere Eigenschaften eines Objekts,
 * das getrackt wird. Sie enthält Methoden zur Merkmalsextraktion und Verwaltung dieser Eigenschaften.
 */

#ifndef PERSON_HPP
#define PERSON_HPP

#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @class Person
 * @brief Repräsentiert eine zu trackende Person mit ihren Merkmalen und Methoden zur Analyse.
 */
class Person {
private:
    int id; ///< Einzigartige ID des Tracks.
    std::vector<cv::Point> contour; ///< Kontur des Objekts.
    cv::Point centroid; ///< Schwerpunkt des Objekts.
    cv::Rect boundingBox; ///< Bounding Box der Kontur.
    double area; ///< Fläche der Kontur.
    double aspectRatio; ///< Verhältnis von Breite zu Höhe der Bounding Box.
    double compactness; ///< Kompaktheit der Kontur als Formparameter.
    std::vector<cv::KeyPoint> keypoints; ///< Keypoints der Kontur.
    cv::Mat hsvHistogram; ///< HSV-Histogramm für Farbmerkmale.
    std::vector<cv::Point2f> trackedPoints; ///< Getrackte Punkte für Optical Flow.

public:
    /**
     * @brief Standardkonstruktor.
     */
    Person();

    /**
     * @brief Konstruktor mit ID und Kontur.
     * @param id Eindeutige ID des Tracks.
     * @param contour Kontur des Objekts.
     */
    Person(int id, const std::vector<cv::Point>& contour);

    /**
     * @brief Destruktor.
     */
    ~Person();

    // Getter und Setter
    /**
     * @brief Gibt die ID der Person zurück.
     * @return ID der Person.
     */
    int getId() const;

    /**
     * @brief Setzt die ID der Person.
     * @param id Eindeutige ID des Tracks.
     */
    void setId(int id);

    /**
     * @brief Gibt die Kontur des Objekts zurück.
     * @return Vektor der Punkte, die die Kontur definieren.
     */
    const std::vector<cv::Point>& getContour() const;

    /**
     * @brief Setzt die Kontur des Objekts und berechnet abhängige Eigenschaften.
     * @param contour Vektor der Punkte, die die Kontur definieren.
     * @param frame Das aktuelle Videoframe.
     */
    void setContour(const std::vector<cv::Point>& contour, const cv::Mat& frame);

    /**
     * @brief Gibt den Schwerpunkt des Objekts zurück.
     * @return Schwerpunkt der Kontur.
     */
    cv::Point getCentroid() const;

    /**
     * @brief Setzt den Schwerpunkt des Objekts.
     * @param centroid Der Schwerpunkt der Kontur.
     */
    void setCentroid(const cv::Point& centroid);

    /**
     * @brief Gibt die Bounding Box der Kontur zurück.
     * @return Rechteck, das die Kontur umgibt.
     */
    cv::Rect getBoundingBox() const;

    /**
     * @brief Gibt die Fläche der Kontur zurück.
     * @return Fläche der Kontur.
     */
    double getArea() const;

    /**
     * @brief Gibt das Verhältnis von Breite zu Höhe der Bounding Box zurück.
     * @return Aspect Ratio der Bounding Box.
     */
    double getAspectRatio() const;

    /**
     * @brief Gibt die Kompaktheit der Kontur zurück.
     * @return Kompaktheitswert der Kontur.
     */
    double getCompactness() const;

    /**
     * @brief Gibt die Keypoints der Kontur zurück.
     * @return Vektor der Keypoints.
     */
    const std::vector<cv::KeyPoint>& getKeypoints() const;

    /**
     * @brief Setzt die Keypoints der Kontur.
     * @param keypoints Vektor der Keypoints.
     */
    void setKeypoints(const std::vector<cv::KeyPoint>& keypoints);

    /**
     * @brief Gibt die getrackten Punkte zurück.
     * @return Vektor der getrackten Punkte.
     */
    const std::vector<cv::Point2f>& getTrackedPoints() const;

    /**
     * @brief Setzt die getrackten Punkte.
     * @param points Vektor der getrackten Punkte.
     */
    void setTrackedPoints(const std::vector<cv::Point2f>& points);

    /**
     * @brief Gibt das HSV-Histogramm zurück.
     * @return HSV-Histogramm der Kontur.
     */
    cv::Mat getHsvHistogram() const;

    /**
     * @brief Setzt das HSV-Histogramm.
     * @param histogram Das HSV-Histogramm.
     */
    void setHsvHistogram(const cv::Mat& histogram);

    // Methoden zur Merkmalsextraktion
    /**
     * @brief Berechnet die Eigenschaften der Kontur (Fläche, Aspect Ratio, Kompaktheit).
     */
    void calculateProperties();

    /**
     * @brief Extrahiert Keypoints basierend auf der Kontur.
     * @param contour Vektor der Punkte, die die Kontur definieren.
     */
    void extractKeypoints(const std::vector<cv::Point>& contour);

    /**
     * @brief Berechnet das HSV-Histogramm für die Kontur.
     * @param frame Das aktuelle Videoframe.
     */
    void computeHsvHistogram(const cv::Mat& frame);
};

#endif // PERSON_HPP
