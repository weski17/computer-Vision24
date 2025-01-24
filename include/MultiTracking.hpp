/**
 * @file MultiTracking.hpp
 * @brief Header-Datei für die MultiTracking-Klasse zur Verarbeitung und Verfolgung von Personen in Videoframes.
 *
 * Diese Klasse enthält die Hauptpipeline für Multi-Object-Tracking, Visualisierung,
 * und FPS-Berechnung sowie Hilfsfunktionen für Optical Flow, Kalman-Filter, und Hintergrundsubtraktion.
 */

#ifndef MULTITRACKING_HPP
#define MULTITRACKING_HPP

#include <opencv2/opencv.hpp>
#include "Person.hpp"
#include "TrackingPipeline.hpp"
#include <vector>
#include <map>
#include <chrono>
#include <filesystem>

constexpr int MIN_TRACKED_POINTS = 60;      ///< Mindestanzahl an verfolgten Punkten.
constexpr double MIN_AREA = 4500.0;         ///< Mindestfläche für relevante Konturen.

/**
 * @class MultiTracking
 * @brief Klasse für Multi-Object-Tracking in Videoframes.
 */
class MultiTracking {
public:
    /**
     * @brief Konstruktor. Initialisiert die Tracking-Klasse.
     */
    MultiTracking();

    /**
     * @brief Destruktor. Gibt Ressourcen frei.
     */
    ~MultiTracking();

    /**
     * @brief Hauptpipeline für die Verarbeitung eines Frames.
     * @param frame Das aktuelle Videoframe.
     */
    void processFrame(const cv::Mat& frame);

    /**
     * @brief Visualisiert die Tracking-Ergebnisse.
     * @param frame Das aktuelle Videoframe.
     */
    void visualize(const cv::Mat& frame) const;

    /**
     * @brief Berechnet und misst die Frames per Second (FPS).
     * @param frame Das aktuelle Videoframe.
     */
    void measureFPS(const cv::Mat& frame);

private:
    std::map<int, Person> tracks;           ///< Alle aktiven Tracks.
    int nextId;                             ///< Nächste verfügbare ID für neue Tracks.
    cv::Mat prevGrayFrame;                  ///< Vorheriges Graustufen-Frame für Optical Flow.
    TrackingPipeline trackingPipeline;      ///< Objekt der TrackingPipeline für spezifische Tracking-Logik.
    std::map<int, cv::KalmanFilter> kalmanFilters; ///< Kalman-Filter für jeden Track.

    /**
     * @brief Implementiert den ungarischen Algorithmus für die Zuordnung von Konturen zu Tracks.
     * @param costMatrix Kostenmatrix für die Zuordnung.
     * @return Ein Vektor mit den Zuordnungen.
     */
    std::vector<int> hungarianAlgorithm(const std::vector<std::vector<double>>& costMatrix);

    /**
     * @brief Vergleicht zwei HSV-Histogramme.
     * @param hist1 Erstes Histogramm.
     * @param hist2 Zweites Histogramm.
     * @return Ähnlichkeitswert zwischen den Histogrammen.
     */
    double compareHistograms(const cv::Mat& hist1, const cv::Mat& hist2);

    /**
     * @brief Berechnet das HSV-Histogramm für eine gegebene Kontur.
     * @param frame Das aktuelle Videoframe.
     * @param contour Die zu analysierende Kontur.
     * @return HSV-Histogramm der Kontur.
     */
    static cv::Mat calculateHsvHistogram(const cv::Mat& frame, const std::vector<cv::Point>& contour);

    /**
     * @brief Führt KNN-basierte Hintergrundsubtraktion aus.
     * @param frame Das aktuelle Videoframe.
     * @return Maske nach der Hintergrundsubtraktion.
     */
    cv::Mat applyKnn(const cv::Mat& frame);

    /**
     * @brief Findet Konturen in der gegebenen Maske.
     * @param mask Die Binärmaske.
     * @return Ein Vektor aller gefundenen Konturen.
     */
    std::vector<std::vector<cv::Point>> findContours(const cv::Mat& mask);

    /**
     * @brief Wendet Optical Flow auf die gegebenen Konturen an.
     * @param frame Das aktuelle Videoframe.
     * @param contours Die zu analysierenden Konturen.
     */
    void applyOpticalFlow(const cv::Mat& frame, std::vector<std::vector<cv::Point>>& contours);

    /**
     * @brief Initialisiert einen Kalman-Filter für die gegebene Person.
     * @param person Die zu verfolgene Person.
     */
    void initializeKalmanFilter(Person& person);

    /**
     * @brief Aktualisiert die Zustände aller Kalman-Filter.
     */
    void updateKalmanFilters();

    /**
     * @brief Ordnet erkannte Konturen den bestehenden Tracks zu.
     * @param contours Die zuzuordnenden Konturen.
     * @param frame Das aktuelle Videoframe.
     */
    void assignContoursToTracks(const std::vector<std::vector<cv::Point>>& contours, const cv::Mat& frame);

    /**
     * @brief Übergibt die verarbeiteten Daten an die Spiellogik.
     */
    void passDataToGameLogic();
};

#endif // MULTITRACKING_HPP
