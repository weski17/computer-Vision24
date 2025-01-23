#ifndef MULTITRACKING_HPP
#define MULTITRACKING_HPP

#include <opencv2/opencv.hpp>
#include "Person.hpp"
#include "TrackingPipeline.hpp"
#include <vector>
#include <map>
#include <chrono>

constexpr int MIN_TRACKED_POINTS = 60;
constexpr double MIN_AREA = 4500.0; // Mindestfläche für relevante Konturen

class MultiTracking {
public:
    MultiTracking();
    ~MultiTracking();

    void processFrame(const cv::Mat& frame); // Hauptpipeline
    void visualize(const cv::Mat& frame) const;   // Visualisierung der Ergebnisse
    void measureFPS(const cv::Mat& frame); // FPS-Berechnung

private:
    std::map<int, Person> tracks;           // Alle Tracks
    int nextId;                             // Nächste verfügbare ID
    cv::Mat prevGrayFrame;                  // Vorheriges Graustufen-Frame für Optical Flow
    TrackingPipeline trackingPipeline;      // Objekt der TrackingPipeline
    std::map<int, cv::KalmanFilter> kalmanFilters;

    std::vector<int> hungarianAlgorithm(const std::vector<std::vector<double>>& costMatrix);
    
    // Hilfsmethoden
    double compareHistograms(const cv::Mat& hist1, const cv::Mat& hist2);
    static cv::Mat calculateHsvHistogram(const cv::Mat& frame, const std::vector<cv::Point>& contour);
   
    cv::Mat applyKnn(const cv::Mat& frame);                            // Hintergrundsubtraktion
    std::vector<std::vector<cv::Point>> findContours(const cv::Mat& mask); // Konturenerkennung
    void applyOpticalFlow(const cv::Mat& frame, std::vector<std::vector<cv::Point>> contours);                       // Optical Flow
    void initializeKalmanFilter(Person& person);                      // Kalman-Filter initialisieren
    void updateKalmanFilters();                                       // Kalman-Filter aktualisieren
    void assignContoursToTracks(const std::vector<std::vector<cv::Point>>& contours, const cv::Mat& frame); // Konturen zuordnen
    void passDataToGameLogic();                                       // Datenübergabe an Spiellogik
};

#endif // MULTITRACKING_HPP
