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
#include <unordered_map>
#include <tinyxml2.h>
#include <algorithm>

constexpr int MIN_TRACKED_POINTS = 60;      ///< Mindestanzahl an verfolgten Punkten.
constexpr double MIN_AREA = 4500.0;         ///< Mindestfläche für relevante Konturen.

/**
 * @struct BoundingBox
 * @brief Struktur zur Darstellung einer Bounding Box in einem Frame.
 *
 * Enthält Informationen zur Position und Größe der Bounding Box sowie die zugehörige Frame- und Objekt-ID.
 */
struct BoundingBox {
    int frame;          ///< Frame-Nummer
    int id;             ///< ID der Bounding Box
    float xtl, ytl;     ///< Obere linke Ecke (x, y)
    float xbr, ybr;     ///< Untere rechte Ecke (x, y)

    /**
     * @brief Konstruktor für die BoundingBox-Struktur.
     * @param f Frame-Nummer
     * @param i ID der Bounding Box
     * @param x1 Obere linke Ecke x-Koordinate
     * @param y1 Obere linke Ecke y-Koordinate
     * @param x2 Untere rechte Ecke x-Koordinate
     * @param y2 Untere rechte Ecke y-Koordinate
     */
    BoundingBox(int f, int i, float x1, float y1, float x2, float y2)
        : frame(f), id(i), xtl(x1), ytl(y1), xbr(x2), ybr(y2) {}

    /**
     * @brief Berechnet die Fläche der Bounding Box.
     * @return Fläche der Bounding Box.
     */
    float area() const {
        return std::max(0.0f, xbr - xtl) * std::max(0.0f, ybr - ytl);
    }
};

/**
 * @class MultiTracking
 * @brief Klasse für Multi-Object-Tracking in Videoframes.
 *
 * Diese Klasse implementiert die Hauptpipeline für die Verarbeitung von Videoframes, 
 * einschließlich Hintergrundsubtraktion, Konturenerkennung, Zuordnung von Konturen zu Tracks, 
 * Anwendung von Optical Flow, Aktualisierung von Kalman-Filtern, und Berechnung von Tracking-Metriken.
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
     *
     * Führt alle Schritte der Tracking-Pipeline aus, einschließlich Hintergrundsubtraktion,
     * Konturenerkennung, Zuordnung von Konturen zu Tracks, Anwendung von Optical Flow,
     * Aktualisierung von Kalman-Filtern, Datenübergabe an die Spiellogik, Visualisierung
     * und FPS-Messung.
     */
    void processFrame(const cv::Mat& frame);

    /**
     * @brief Visualisiert die Tracking-Ergebnisse.
     * @param frame Das aktuelle Videoframe.
     *
     * Zeichnet Bounding Boxes, IDs, Centroids und Keypoints für jede verfolgte Person auf das Frame.
     * Speichert das visualisierte Video und zeigt es in einem OpenCV-Fenster an.
     */
    void visualize(const cv::Mat& frame) const;

    /**
     * @brief Berechnet und misst die Frames per Second (FPS).
     * @param frame Das aktuelle Videoframe.
     *
     * Verwendet die Zeitstempel der Frames, um die aktuelle FPS zu berechnen und gibt diese
     * alle 1 Sekunde aus.
     */
    void measureFPS(const cv::Mat& frame);

    /**
     * @brief Lädt die Ground-Truth-Daten aus einer XML-Datei.
     * @param xmlFilePath Pfad zur XML-Datei mit den Ground-Truth-Daten.
     * @param frameCounter Referenz auf den Frame-Zähler (wird nicht verwendet und kann entfernt werden).
     *
     * Verwendet TinyXML2, um die XML-Datei zu laden und speichert die Bounding Boxes für jeden Frame
     * in der `groundTruthData`-Map.
     */
    void loadGroundTruth(const std::string& xmlFilePath, int& frameCounter);

    /**
     * @brief Berechnet die Intersection over Union (IoU) zwischen zwei Bounding Boxes.
     * @param box1 Erste Bounding Box.
     * @param box2 Zweite Bounding Box.
     * @return IoU-Wert zwischen den beiden Bounding Boxes.
     */
    float calculateIoU(const BoundingBox& box1, const BoundingBox& box2) const;

    /**
     * @brief Testet die IoU-Berechnung zwischen getrackten Bounding Boxes und Ground-Truth-Daten.
     *
     * Durchläuft alle Frames und berechnet die IoU zwischen den Ground-Truth-Boxen und den vom Tracking-System
     * generierten Boxen. Gibt die Ergebnisse und die durchschnittliche IoU aus.
     */
    void testIoU();

    /**
     * @brief Berechnet die Multiple Object Tracking Accuracy (MOTA).
     * @return Der berechnete MOTA-Wert.
     *
     * MOTA berücksichtigt False Negatives (FN), False Positives (FP) und ID-Switches (IDSW) 
     * im Verhältnis zu den Ground-Truth-Objekten.
     */
    float calculateMOTA();

    /**
     * @brief Berechnet den F1-Score für das Tracking.
     * @return Der berechnete F1-Score.
     *
     * Der F1-Score ist das harmonische Mittel von Präzision und Recall und gibt eine ausgewogene
     * Metrik für die Tracking-Performance.
     */
    float calculateF1Score();

    /**
     * @brief Berechnet die Mean Average Precision (mAP) für das Tracking.
     * @return Der berechnete mAP-Wert.
     *
     * Die mAP ist ein Durchschnitt der Average Precision (AP) über alle Klassen oder Objekte.
     */
    float calculateMeanAveragePrecision();

private:
    std::unordered_map<int, std::vector<BoundingBox>> groundTruthData; ///< Ground-Truth-Daten: Frame -> Bounding Boxes

    std::map<int, Person> tracks;           ///< Alle aktiven Tracks, zugeordnet nach Track-ID.
    int nextId;                             ///< Nächste verfügbare ID für neue Tracks.
    cv::Mat prevGrayFrame;                  ///< Vorheriges Graustufen-Frame für Optical Flow.
    TrackingPipeline trackingPipeline;      ///< Objekt der TrackingPipeline für spezifische Tracking-Logik.
    std::map<int, cv::KalmanFilter> kalmanFilters; ///< Kalman-Filter für jeden Track.

    /**
     * @brief Implementiert den ungarischen Algorithmus für die Zuordnung von Konturen zu Tracks.
     * @param costMatrix Kostenmatrix für die Zuordnung.
     * @return Ein Vektor mit den Zuordnungen, wobei der Index dem Track entspricht und der Wert die zugeordnete Kontur ist.
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
     * @return Ein Vektor aller gefundenen, gefilterten Konturen.
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
     * @param person Die zu verfolgende Person.
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
