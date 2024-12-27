#ifndef BACKGROUNDSUBTRACTIONPIPELINE_HPP
#define BACKGROUNDSUBTRACTIONPIPELINE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>
#include <iostream>

// Struktur zur Speicherung von Evaluationsmetriken
struct Metrics {
    double precision;  ///< Präzision der Vorhersage
    double recall;     ///< Trefferquote (Recall)
    double f1;         ///< F1-Score (harmonisches Mittel aus Präzision und Recall)
    double accuracy;   ///< Genauigkeit
    double iou;        ///< Intersection over Union (IoU) für die Überlappung
};

// Klasse für die Hintergrundsubtraktions-Pipeline
class BackgroundSubtractionPipeline {
public:
    BackgroundSubtractionPipeline();  ///< Konstruktor: Initialisiert die Hintergrundsubtraktoren und andere Variablen

    /**
     * @brief Initialisiert die Hintergrundsubtraktoren (MOG2 und KNN) mit den notwendigen Einstellungen.
     */
    void initializeBackgroundSubtractor();

    /**
     * @brief Wendet Vorverarbeitung auf das Eingabebild an, inklusive Rauschreduktion und Kontrastanpassung.
     * @param frame Eingabebild im BGR-Format
     * @return Vorverarbeitetes Bild im BGR-Format
     */
    cv::Mat applyPreprocessing(const cv::Mat& frame);

    /**
     * @brief Wendet den MOG2-Algorithmus (Mixture of Gaussians) auf das Eingabebild an.
     * @param frame Eingabebild im BGR-Format
     * @return Binäre Maske, die den Vordergrund darstellt
     */
    cv::Mat applyMixtureOfGaussians(const cv::Mat& frame);

    /**
     * @brief Wendet den KNN-Algorithmus auf das Eingabebild an.
     * @param frame Eingabebild im BGR-Format
     * @return Binäre Maske, die den Vordergrund darstellt
     */
    cv::Mat applyKNN(const cv::Mat& frame);

    /**
     * @brief Führt benutzerdefinierte Hintergrundsubtraktion durch und erstellt eine Vordergrundmaske.
     * @param frame Eingabebild im BGR-Format
     * @return Binäre Maske mit erkannten Vordergrundbereichen
     */
    cv::Mat IntervalBGSubtraction(const cv::Mat& frame);

    /**
     * @brief Verbessert die Qualität einer Eingabemaske durch Filterung und Morphologie-Operationen.
     * @param inputMask Binäre Maske, die verbessert werden soll
     * @return Verbesserte binäre Maske
     */
    cv::Mat improveMask(const cv::Mat& inputMask);

    /**
     * @brief Lädt den ersten Frame eines Videos als Ground-Truth-Maske.
     * @param videoPath Dateipfad zum Video
     * @return Ground-Truth-Maske als binäre Maske
     */
    cv::Mat loadFirstFrameAsGroundTruth(const std::string& videoPath);

    /**
     * @brief Berechnet Metriken zur Bewertung der Genauigkeit der Masken.
     * @param predictedMask Berechnete Maske der Hintergrundsubtraktion
     * @param groundTruthMask Ground-Truth-Maske zur Bewertung
     * @return Struktur mit berechneten Metriken
     */
    Metrics calculateMetrics(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask);

    /**
     * @brief Berechnet und zeigt Metriken für verschiedene Hintergrundsubtraktionsmethoden an.
     * @param frame Eingabeframe im BGR-Format
     * @param groundTruthMask Die Ground-Truth-Maske zur Bewertung
     */
    void displayMetricsForMethods(const cv::Mat& frame, const cv::Mat& groundTruthMask);

    /**
     * @brief Speichert die Ergebnisse der Hintergrundsubtraktion in Videos und zeigt Metriken an.
     * @param frame Eingabeframe im BGR-Format
     * @param groundTruthMask Ground-Truth-Maske zur Bewertung
     */
    void saveBackgroundSubtractionResults(const cv::Mat& frame, const cv::Mat& groundTruthMask);

    /**
     * @brief Speichert den ersten Frame eines Videos als PNG-Bilddatei.
     * @param videoPath Pfad zum Video
     * @param outputPath Pfad zum Verzeichnis für das Bild
     */
    void saveFirstFrameAsPNG(const std::string& videoPath, const std::string& outputPath);

    /**
     * @brief Initialisiert VideoWriter für die Speicherung von Ergebnissen und setzt die Videoeigenschaften.
     * @param outputDirectory Verzeichnis für die Videoausgabe
     * @param cap VideoCapture-Objekt, das die Eingabequelle repräsentiert
     */
    void initializeVideoWriters(const std::string &outputDirectory, cv::VideoCapture &cap);

    /**
     * @brief Gibt VideoWriter-Ressourcen frei.
     */
    void releaseVideoWriters();

    /**
     * @brief Initialisiert Videoquelle und lädt die Ground-Truth-Maske.
     * @param videoPath Pfad zum Video
     * @param cap VideoCapture-Objekt
     * @param groundTruthMask Referenz zur Ground-Truth-Maske
     * @param outputPath Pfad für die Ausgabe
     * @return Erfolg (true) oder Fehler (false) bei der Initialisierung
     */
    bool initializeVideoAndLoadGroundTruth(const std::string& videoPath, cv::VideoCapture& cap, cv::Mat& groundTruthMask, const std::string& outputPath);

private:
    cv::VideoWriter outputVideoMOG2;      ///< VideoWriter für MOG2-Ergebnisse
    cv::VideoWriter outputVideoKNN;       ///< VideoWriter für KNN-Ergebnisse
    cv::VideoWriter outputVideoCodebook;  ///< VideoWriter für Codebook-Ergebnisse

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2; ///< MOG2-Algorithmus für Hintergrundsubtraktion
    cv::Ptr<cv::BackgroundSubtractorKNN> knn;   ///< KNN-Algorithmus für Hintergrundsubtraktion

    cv::Mat minInterval;  ///< Minimales Intervall für Hintergrundmodell
    cv::Mat maxInterval;  ///< Maximales Intervall für Hintergrundmodell
    int frameCount;       ///< Zähler für die Anzahl verarbeiteter Frames
};

#endif // BACKGROUNDSUBTRACTIONPIPELINE_HPP
