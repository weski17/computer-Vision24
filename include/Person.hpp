#ifndef PERSON_HPP
#define PERSON_HPP

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class Person
 * @brief Repräsentiert eine verfolgte Person mit ID, Kontur und Schwerpunkt (Centroid).
 */
class Person {
public:
    /**
     * @brief Standardkonstruktor.
     * Initialisiert die ID mit -1 und den Schwerpunkt (Centroid) mit (0, 0).
     */
    Person();

    /**
     * @brief Konstruktor mit ID und Kontur.
     * @param id Eindeutige ID der Person.
     * @param contour Kontur der Person, um den Schwerpunkt zu berechnen.
     */
    Person(int id, const std::vector<cv::Point>& contour);

    /**
     * @brief Destruktor.
     */
    ~Person();

    /**
     * @brief Gibt die ID der Person zurück.
     * @return Die ID der Person.
     */
    int getId() const;

    /**
     * @brief Setzt die ID der Person.
     * @param id Neue ID.
     */
    void setId(int id);

    /**
     * @brief Gibt die Kontur der Person zurück.
     * @return Die Kontur als Referenz auf einen Vektor von Punkten.
     */
    const std::vector<cv::Point>& getContour() const;

    /**
     * @brief Setzt die Kontur der Person und berechnet den Schwerpunkt.
     * @param contour Neue Kontur der Person.
     */
    void setContour(const std::vector<cv::Point>& contour);

    /**
     * @brief Gibt den Schwerpunkt (Centroid) der Person zurück.
     * @return Der Schwerpunkt als `cv::Point`.
     */
    cv::Point getCentroid() const;

    /**
     * @brief Setzt den Schwerpunkt (Centroid) der Person.
     * @param centroid Neuer Schwerpunkt.
     */
    void setCentroid(const cv::Point& centroid);

private:
    int id; ///< Eindeutige ID der Person.
    std::vector<cv::Point> contour; ///< Kontur der Person.
    cv::Point centroid; ///< Schwerpunkt (Centroid) der Person basierend auf der Kontur.
};

#endif // PERSON_HPP
