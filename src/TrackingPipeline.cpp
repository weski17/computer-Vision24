#include "TrackingPipeline.hpp"
#include <queue>
#include <fstream>
#include <cmath>
#include <numeric>

// Globale Variablen für Ground-Truth-Generierung
static cv::Mat prevFrame; // Vorheriger Frame
static std::vector<std::vector<cv::Point>> groundTruthContours;

// Konstruktor
TrackingPipeline::TrackingPipeline()
    : trackedPerson(-1, std::vector<cv::Point>()), isTracking(false)
{
    backgroundSubtraction = BackgroundSubtractionPipeline();
    initializeKalmanFilter(); // Initialisierung des Kalman-Filters
}

// Destruktor
TrackingPipeline::~TrackingPipeline() {}

// Kalman-Filter initialisieren
void TrackingPipeline::initializeKalmanFilter()
{
    kalmanFilter = cv::KalmanFilter(4, 2, 0);
    kalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                                     0, 1, 0, 1,
                                     0, 0, 1, 0,
                                     0, 0, 0, 1);
    kalmanFilter.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);
    kalmanFilter.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2;
    kalmanFilter.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1e-1;
    kalmanFilter.errorCovPost = cv::Mat::eye(4, 4, CV_32F);
}

// Positionsfehler berechnen
double TrackingPipeline::computePositionError(const cv::Point &predictedCentroid, const cv::Point &groundTruthCentroid)
{
    return cv::norm(predictedCentroid - groundTruthCentroid);
}

// F-Score berechnen
double TrackingPipeline::calculateFScore(double precision, double recall)
{
    return (2 * precision * recall) / (precision + recall);
}

// Tracking evaluieren
void TrackingPipeline::evaluateTrackingResults()
{
    if (groundTruthContours.empty() || trackedPerson.getContour().empty())
    {
        return;
    }

    // Berechne Bounding-Boxes
    cv::Rect predictedBox = cv::boundingRect(trackedPerson.getContour());
    cv::Rect groundTruthBox = cv::boundingRect(groundTruthContours[0]);

    // Zentren der Bounding-Boxes berechnen
    cv::Point predictedCentroid((predictedBox.tl() + predictedBox.br()) / 2);
    cv::Point groundTruthCentroid((groundTruthBox.tl() + groundTruthBox.br()) / 2);

    // Positionsfehler berechnen
    double pe = computePositionError(predictedCentroid, groundTruthCentroid);
    peValues.push_back(pe);

    // Präzision, Recall und F-Score berechnen
    double intersectionArea = (predictedBox & groundTruthBox).area();
    double precision = intersectionArea / predictedBox.area();
    double recall = intersectionArea / groundTruthBox.area();
    double fScore = calculateFScore(precision, recall);

    // Ergebnisse speichern
    fScores.push_back(fScore);
}

// Ground Truth generieren
void TrackingPipeline::generateGroundTruth(const cv::Mat &frame)
{
    cv::Mat fgMask = applyKnn(frame);

    // Konturen finden
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        return;
    }

    // Größte gültige Kontur bestimmen
    std::vector<cv::Point> largestContour;
    double maxArea = 2000.0; // Mindestfläche
    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area > maxArea)
        {
            maxArea = area;
            largestContour = contour;
        }
    }

    if (!largestContour.empty())
    {
        groundTruthContours.push_back(largestContour);
    }
}

// Hintergrundsubtraktion
cv::Mat TrackingPipeline::applyKnn(const cv::Mat &frame)
{
    cv::Mat preprocessed = backgroundSubtraction.applyPreprocessing(frame);
    cv::Mat knn = backgroundSubtraction.applyKNN(preprocessed);
    return backgroundSubtraction.improveMask(knn);
}

void TrackingPipeline::detectAndTrackPersons(const cv::Mat& frame) {

    cv::Mat fgMask = applyKnn(frame);

    // Konturen finden
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Größte gültige Kontur bestimmen
    std::vector<cv::Point> largestContour;
    double maxArea = 2000.0;
    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area > maxArea)
        {
            maxArea = area;
            largestContour = contour;
        }
    }

    if (!largestContour.empty())
    {
        isTracking = true;

        // Keypoints aus der größten Kontur extrahieren
        extractKeypoints(largestContour);

        // Optical Flow anwenden, um Keypoints zu verfeinern
        applyOpticalFlow(frame);

        // Kalman-Filter aktualisieren und neue Position schätzen
        if (!trackedPoints.empty()) {

            cv::Point2f meanPoint(0, 0);
            for (const auto &point : trackedPoints)
            {
                meanPoint += point;
            }
            meanPoint.x /= trackedPoints.size();
            meanPoint.y /= trackedPoints.size();

            // Kalman-Filter aktualisieren
            updateKalmanFilter(cv::Point(static_cast<int>(meanPoint.x), static_cast<int>(meanPoint.y)));
        }

        // Nach der Verarbeitung die Kontur speichern
        trackedPerson.setContour(largestContour,frame);
    } else {

        isTracking = false;
        trackedPoints.clear();
        trackedPerson.setContour(std::vector<cv::Point>(), frame);
    }
}

// Keypoints aus Kontur extrahieren
void TrackingPipeline::extractKeypoints(const std::vector<cv::Point> &contour)
{
    std::vector<cv::Point2f> contourKeypoints;
    int step = std::max(1, static_cast<int>(contour.size() / 100));
    for (size_t i = 0; i < contour.size(); i += step)
    {
        contourKeypoints.push_back(contour[i]);
    }
    trackedPoints = contourKeypoints;
}

// Optical Flow anwenden
void TrackingPipeline::applyOpticalFlow(const cv::Mat &frame)
{
    if (!prevGrayFrame.empty() && !trackedPoints.empty())
    {
        std::vector<uchar> status;
        std::vector<float> err;
        cv::Mat nextGray;
        cv::cvtColor(frame, nextGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> nextPoints;
        cv::calcOpticalFlowPyrLK(prevGrayFrame, nextGray, trackedPoints, nextPoints, status, err);

        std::vector<cv::Point2f> validPoints;
        for (size_t i = 0; i < trackedPoints.size(); ++i)
        {
            if (status[i])
            {
                validPoints.push_back(nextPoints[i]);
            }
        }

        trackedPoints = validPoints;
        prevGrayFrame = nextGray.clone();
    }
    else
    {
        cv::cvtColor(frame, prevGrayFrame, cv::COLOR_BGR2GRAY);
    }
}

// Kalman-Filter aktualisieren
void TrackingPipeline::updateKalmanFilter(const cv::Point &measuredPoint)
{
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << measuredPoint.x, measuredPoint.y);
    kalmanFilter.correct(measurement);
    cv::Mat prediction = kalmanFilter.predict();
    cv::Point predictedPoint(prediction.at<float>(0), prediction.at<float>(1));
    trackedPerson.setCentroid(predictedPoint);
}

// Frame verarbeiten
void TrackingPipeline::processFrame(cv::Mat& frame) {
    static cv::VideoWriter writer;
    static bool isWriterInitialized = false;
    const std::string outputPath = "data/output/video2.avi";

    // VideoWriter initialisieren
    if (!isWriterInitialized) {
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        double fps = 30;
        cv::Size frameSize(frame.cols, frame.rows);

        writer.open(outputPath, codec, fps, frameSize, true);
        if (!writer.isOpened()) {
            return;
        }
        isWriterInitialized = true;
    }

    // Tracking- und Visualisierungslogik
    
    detectAndTrackPersons(frame);
    visualizeResults(frame);
}

//keine Visuallisereung mehr
void TrackingPipeline::visualizeResults(cv::Mat &frame)
{
    if (!isTracking)
    {
        return; // Keine Visualisierung, wenn kein Tracking aktiv ist
    }

    // Die grüne Kontur wird nicht mehr gezeichnet
    const std::vector<cv::Point> &contour = trackedPerson.getContour();
    if (!contour.empty())
    {
        // Entfernt: cv::drawContours(frame, {contour}, -1, cv::Scalar(0, 255, 0), 2);
        // Die Kontur bleibt intern verfügbar für das Tracking
    }
}

// Frame evaluieren
void TrackingPipeline::evaluateFrame(cv::Mat &frame)
{
    if (!groundTruthContours.empty())
    {
        evaluateTrackingResults();
    }
}

// Kontur der verfolgten Person abrufen
const std::vector<cv::Point> &TrackingPipeline::getTrackedContour() const
{
    return trackedPerson.getContour();
}
