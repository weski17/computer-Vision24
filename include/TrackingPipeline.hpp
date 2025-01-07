#ifndef TRACKING_PIPELINE_HPP
#define TRACKING_PIPELINE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "BackgroundSubtractionPipeline.hpp"
#include "Person.hpp"

/**
 * @class TrackingPipeline
 * @brief Pipeline zur Objektnachverfolgung und -bewertung.
 */
class TrackingPipeline {
public:
    /**
     * @brief Konstruktor der Klasse.
     */
    TrackingPipeline();

    /**
     * @brief Destruktor der Klasse.
     */
    ~TrackingPipeline();

    /**
     * @brief Verarbeitet einen einzelnen Frame.
     * @param frame Der zu verarbeitende Frame.
     */
    void processFrame(cv::Mat& frame);

    /**
     * @brief Generiert Ground Truth für die Nachverfolgung.
     * @param frame Eingabeframe zur Ground-Truth-Erstellung.
     */
    void generateGroundTruth(const cv::Mat& frame);

    /**
     * @brief Führt Hintergrundsubtraktion mittels KNN durch.
     * @param frame Eingabeframe.
     * @return Ergebnisframe nach Subtraktion.
     */
    cv::Mat applyKnn(const cv::Mat& frame);

private:

    /**
     * @brief Erkennt und verfolgt Personen im Frame.
     * @param frame Eingabeframe.
     */
    void detectAndTrackPersons(const cv::Mat& frame);

    /**
     * @brief Initialisiert den Kalman-Filter.
     */
    void initializeKalmanFilter();

    /**
     * @brief Extrahiert Keypoints aus einer Kontur.
     * @param contour Eingabekontur.
     */
    void extractKeypoints(const std::vector<cv::Point>& contour);

    /**
     * @brief Wendet optischen Fluss auf den Frame an.
     * @param frame Eingabeframe.
     */
    void applyOpticalFlow(const cv::Mat& frame);

    /**
     * @brief Aktualisiert den Kalman-Filter mit gemessenen Daten.
     * @param measuredPoint Gemessener Punkt.
     */
    void updateKalmanFilter(const cv::Point& measuredPoint);

    /**
     * @brief Bewertet die Tracking-Ergebnisse.
     */
    void evaluateTrackingResults();

    /**
     * @brief Berechnet die Positionsabweichung.
     * @param predictedCentroid Vorhergesagter Schwerpunkt.
     * @param groundTruthCentroid Ground-Truth-Schwerpunkt.
     * @return Positionsabweichung.
     */
    double computePositionError(const cv::Point& predictedCentroid, const cv::Point& groundTruthCentroid);

    /**
     * @brief Berechnet den F-Score.
     * @param precision Präzision.
     * @param recall Rückrufrate.
     * @return F-Score.
     */
    double calculateFScore(double precision, double recall);

    /**
     * @brief Visualisiert die Ergebnisse auf dem Frame.
     * @param frame Der zu visualisierende Frame.
     */
    void visualizeResults(cv::Mat& frame);

    /**
     * @brief Bewertet den aktuellen Frame.
     * @param frame Eingabeframe.
     */
    void evaluateFrame(cv::Mat& frame);

    BackgroundSubtractionPipeline backgroundSubtraction; ///< Pipeline zur Hintergrundsubtraktion.
    Person trackedPerson; ///< Verfolgte Person.
    cv::KalmanFilter kalmanFilter; ///< Kalman-Filter für die Nachverfolgung.
    cv::Mat prevGrayFrame; ///< Vorheriger Graustufen-Frame.
    std::vector<cv::Point2f> groundingtruth; ///< Ground-Truth-Punkte.
    std::vector<cv::Point2f> trackedPoints; ///< Verfolgte Punkte.
    bool isTracking; ///< Status, ob Tracking aktiv ist.

    std::vector<double> fScores; ///< Liste der F-Scores.
    std::vector<double> peValues; ///< Liste der Positionsabweichungen.
};

#endif // TRACKING_PIPELINE_HPP
