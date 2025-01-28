/**
 * @file MultiTracking.cpp
 * @brief Implementierung der MultiTracking-Klasse für Multi-Object-Tracking.
 *
 * Diese Datei enthält die Implementierung der MultiTracking-Klasse, die für die
 * Verarbeitung von Videoframes, Hintergrundsubtraktion, Konturenerkennung, Zuordnung
 * von Konturen zu Tracks, Anwendung von Optical Flow und Aktualisierung von Kalman-Filtern
 * verantwortlich ist. Zudem werden verschiedene Tracking-Metriken berechnet.
 */

#include "MultiTracking.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <limits>
#include <iostream>
#include <chrono>
#include <filesystem>

// Konstruktor
/**
 * @brief Konstruktor der MultiTracking-Klasse.
 *
 * Initialisiert die MultiTracking-Klasse mit der nächsten verfügbaren ID auf 0
 * und erstellt ein TrackingPipeline-Objekt.
 */
MultiTracking::MultiTracking() : nextId(0), trackingPipeline() {}

// Destruktor
/**
 * @brief Destruktor der MultiTracking-Klasse.
 *
 * Gibt alle Ressourcen frei, die von der MultiTracking-Klasse verwendet werden.
 */
MultiTracking::~MultiTracking() {}

/**
 * @brief Initialisiert den Kalman-Filter für eine gegebene Person.
 *
 * Erstellt und konfiguriert einen Kalman-Filter für die gegebene Person basierend auf deren
 * aktuellem Mittelpunkt (Centroid). Der Kalman-Filter wird dann in der `kalmanFilters`-Map
 * mit der entsprechenden Personen-ID gespeichert.
 *
 * @param person Die zu verfolgende Person, für die der Kalman-Filter initialisiert wird.
 */
void MultiTracking::initializeKalmanFilter(Person& person) {
    cv::KalmanFilter kalman(6, 2, 0); // Zustand: (x, y, dx, dy, ddx, ddy)
    cv::Point centroid = person.getCentroid();

    // Initialisieren des Zustands
    kalman.statePre.at<float>(0) = centroid.x;
    kalman.statePre.at<float>(1) = centroid.y;
    kalman.statePre.at<float>(2) = 0; // dx
    kalman.statePre.at<float>(3) = 0; // dy
    kalman.statePre.at<float>(4) = 0; // ddx
    kalman.statePre.at<float>(5) = 0; // ddy

    // Übergangsmatrix (inkl. Beschleunigung)
    kalman.transitionMatrix = (cv::Mat_<float>(6, 6) <<
        1, 0, 1, 0, 0.5, 0,
        0, 1, 0, 1, 0, 0.5,
        0, 0, 1, 0, 1, 0,
        0, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);

    // Messmatrix
    kalman.measurementMatrix = (cv::Mat_<float>(2, 6) <<
        1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0);

    // Prozess- und Messrauschen
    setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-4));
    setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kalman.errorCovPost, cv::Scalar::all(1));

    // Kalman-Filter speichern
    kalmanFilters[person.getId()] = kalman;
}

/**
 * @brief Führt KNN-basierte Hintergrundsubtraktion auf einem Frame durch.
 *
 * Verwendet die TrackingPipeline, um die Hintergrundsubtraktion mittels KNN (K-Nearest Neighbors)
 * durchzuführen und gibt die resultierende Maske zurück.
 *
 * @param frame Das aktuelle Videoframe.
 * @return Die Maske nach der Hintergrundsubtraktion.
 */
cv::Mat MultiTracking::applyKnn(const cv::Mat& frame) {
    return trackingPipeline.applyKnn(frame);
}

/**
 * @brief Findet und filtert Konturen in einer gegebenen Maske basierend auf der Mindestfläche.
 *
 * Verwendet OpenCV-Funktionen, um alle externen Konturen in der Maske zu finden und filtert
 * diese basierend auf der definierten Mindestfläche (`MIN_AREA`), um irrelevante kleine Konturen
 * zu entfernen.
 *
 * @param mask Die Binärmaske, in der Konturen gesucht werden.
 * @return Ein Vektor von gefilterten Konturen.
 */
std::vector<std::vector<cv::Point>> MultiTracking::findContours(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> allContours;
    std::vector<std::vector<cv::Point>> filteredContours;
    
    // Finde alle Konturen
    cv::findContours(mask, allContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Filtere kleine Konturen basierend auf ihrer Fläche
    for (const auto& contour : allContours) {
        if (cv::contourArea(contour) >= MIN_AREA) {
            filteredContours.push_back(contour);
        }
    }

    return filteredContours; // Rückgabe nur der gefilterten Konturen
}

/**
 * @brief Ordnet erkannte Konturen den bestehenden Tracks zu und aktualisiert diese.
 *
 * Erstellt eine Kostenmatrix basierend auf Distanz, Flächendifferenz, Seitenverhältnisunterschied
 * und Histogrammähnlichkeit zwischen bestehenden Tracks und neuen Konturen. Verwendet den
 * ungarischen Algorithmus, um die optimale Zuordnung zu finden. Aktualisiert dann die Tracks
 * entsprechend der Zuordnung und fügt neue Tracks für nicht zugeordnete Konturen hinzu.
 *
 * @param contours Die erkannten Konturen, die den Tracks zugeordnet werden sollen.
 * @param frame Das aktuelle Videoframe.
 */
void MultiTracking::assignContoursToTracks(const std::vector<std::vector<cv::Point>>& contours, const cv::Mat& frame) {
    // Schritt 1: Kostenmatrix erstellen
    std::vector<std::vector<double>> costMatrix(tracks.size(), std::vector<double>(contours.size(), 0));
    int i = 0;

    for (const auto& [trackId, person] : tracks) {
        int j = 0;
        for (const auto& contour : contours) {
            cv::Point trackCentroid = person.getCentroid();
            cv::Moments moments = cv::moments(contour);
            cv::Point contourCentroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

            // Berechnung der Zuordnungskosten
            double distance = cv::norm(trackCentroid - contourCentroid);
            double contourArea = cv::contourArea(contour);
            double areaDifference = std::abs(person.getArea() - contourArea);

            cv::Rect boundingBox = cv::boundingRect(contour);
            double contourAspectRatio = static_cast<double>(boundingBox.width) / boundingBox.height;
            double aspectRatioDifference = std::abs(person.getAspectRatio() - contourAspectRatio);

            cv::Mat contourHistogram = MultiTracking::calculateHsvHistogram(frame, contour);
            double histogramSimilarity = compareHistograms(person.getHsvHistogram(), contourHistogram);

            // Gesamtkosten berechnen
            double totalCost = 0.4 * distance + 0.2 * areaDifference + 0.2 * aspectRatioDifference + 0.2 * (1 - histogramSimilarity);   
            if (totalCost > 80.0) {
                totalCost = 1e9; 
            }

            costMatrix[i][j] = totalCost;

            j++;
        }
        i++;
    }

    // Debugging: Kostenmatrix anzeigen
    std::cout << "Kostenmatrix:" << std::endl;
    for (const auto& row : costMatrix) {
        for (double cost : row) {
            std::cout << cost << " ";
        }
        std::cout << std::endl;
    }

    // Schritt 2: Zuordnung mit Ungarischer Methode
    std::vector<int> assignment = hungarianAlgorithm(costMatrix);

    // Debugging: Zuordnung prüfen
    std::cout << "Zuordnung:" << std::endl;
    for (size_t t = 0; t < assignment.size(); ++t) {
        std::cout << "Track " << t << " -> Contour " << assignment[t] << std::endl;
    }

    // Schritt 3: Tracks aktualisieren
    for (size_t t = 0; t < assignment.size(); ++t) {
        if (assignment[t] != -1) { // Wenn eine gültige Zuordnung existiert
            int contourIdx = assignment[t];
            auto& person = tracks[t];
            person.setContour(contours[contourIdx], frame);
            person.extractKeypoints(contours[contourIdx]);

            // Debugging: Aktualisierte Tracks
            std::cout << "Track " << t << " aktualisiert mit Kontur " << contourIdx << std::endl;
        }
    }

    // Schritt 4: Nicht zugeordnete Konturen als neue Tracks hinzufügen
    std::vector<int> usedContours; // Liste der genutzten Konturen
    for (size_t c = 0; c < contours.size(); ++c) {
        if (std::find(assignment.begin(), assignment.end(), c) == assignment.end()) {
            int newId = nextId++;
            Person newPerson(newId, contours[c]);
            newPerson.extractKeypoints(contours[c]);
            initializeKalmanFilter(newPerson);
            tracks[newId] = newPerson;

            usedContours.push_back(c); // Markiere diese Kontur als verwendet
        }
    }
}

/**
 * @brief Wendet Optical Flow auf die getrackten Punkte der bestehenden Tracks an.
 *
 * Verwendet den Lucas-Kanade-Algorithmus, um die Bewegung der Punkte zwischen dem vorherigen
 * und dem aktuellen Frame zu verfolgen. Aktualisiert die Positionen der Tracks basierend auf
 * den getrackten Punkten oder verwendet den Kalman-Filter zur Vorhersage, falls keine gültigen
 * Punkte vorhanden sind.
 *
 * @param frame Das aktuelle Videoframe.
 * @param contours Die aktuellen Konturen, die für das Tracking verwendet werden.
 */
void MultiTracking::applyOpticalFlow(const cv::Mat& frame,  std::vector<std::vector<cv::Point>>& contours) {
    if (prevGrayFrame.empty()) {
        cv::cvtColor(frame, prevGrayFrame, cv::COLOR_BGR2GRAY);
        return;
    }

    cv::Mat currGrayFrame;
    cv::cvtColor(frame, currGrayFrame, cv::COLOR_BGR2GRAY);

    for (auto& [trackId, person] : tracks) {
        std::vector<cv::Point2f> trackedPoints = person.getTrackedPoints();

        // Wenn zu wenige Punkte vorhanden sind, extrahiere neue Keypoints
        if (trackedPoints.empty() || trackedPoints.size() < MIN_TRACKED_POINTS) {
            // Sicherstellen, dass trackId innerhalb der Konturenliste liegt
            if (trackId >= contours.size()) {
                std::cerr << "Track ID " << trackId << " ist außerhalb des Bereichs der Konturenliste!" << std::endl;
                continue;
            }

            const auto& contour = contours[trackId];
            person.extractKeypoints(contour); // Extrahiere Keypoints aus der Kontur
            trackedPoints = person.getTrackedPoints();

            if (trackedPoints.empty()) {
                // Wenn keine neuen Punkte gefunden werden, Vorhersage durch Kalman-Filter
                if (kalmanFilters.find(trackId) != kalmanFilters.end()) {
                    cv::Mat prediction = kalmanFilters[trackId].predict();
                    cv::Point predictedCentroid(prediction.at<float>(0), prediction.at<float>(1));
                    //person.setCentroid(predictedCentroid);
                } else {
                    std::cerr << "Kalman-Filter für Track ID " << trackId << " nicht gefunden." << std::endl;
                }
                continue; // Überspringe den Optical Flow
            }
        }

        std::vector<cv::Point2f> nextPoints;
        std::vector<uchar> status;
        std::vector<float> err;

        // Optical Flow berechnen
        cv::calcOpticalFlowPyrLK(prevGrayFrame, currGrayFrame, trackedPoints, nextPoints, status, err);

        // Filtere ungültige Punkte
        std::vector<cv::Point2f> validNextPoints;
        for (size_t i = 0; i < nextPoints.size(); ++i) {
            if (status[i]) {
                validNextPoints.push_back(nextPoints[i]);
            }
        }

        // Wenn keine gültigen Punkte vorhanden sind, Vorhersage durch Kalman-Filter
        if (validNextPoints.empty()) {
            if (kalmanFilters.find(trackId) != kalmanFilters.end()) {
                cv::Mat prediction = kalmanFilters[trackId].predict();
                cv::Point predictedCentroid(prediction.at<float>(0), prediction.at<float>(1));
                person.setCentroid(predictedCentroid);
            } else {
                std::cerr << "Kalman-Filter für Track ID " << trackId << " nicht gefunden." << std::endl;
            }
        } else {
            // Aktualisiere getrackte Punkte
            person.setTrackedPoints(validNextPoints);

            // Aktualisiere den Schwerpunkt basierend auf den Keypoints
            cv::Point2f newCentroid(0, 0);
            for (const auto& pt : validNextPoints) {
                newCentroid += pt;
            }
            newCentroid.x /= validNextPoints.size();
            newCentroid.y /= validNextPoints.size();
            person.setCentroid(cv::Point(static_cast<int>(newCentroid.x), static_cast<int>(newCentroid.y)));

            // Aktualisiere den Kalman-Filter mit der neuen Messung
            if (kalmanFilters.find(trackId) != kalmanFilters.end()) {
                cv::Mat measurement(2, 1, CV_32F);
                measurement.at<float>(0) = newCentroid.x;
                measurement.at<float>(1) = newCentroid.y;
                kalmanFilters[trackId].correct(measurement);
            } else {
                std::cerr << "Kalman-Filter für Track ID " << trackId << " nicht gefunden." << std::endl;
            }
        }
    }

    // Aktualisiere den vorherigen Grauwert-Frame
    prevGrayFrame = currGrayFrame.clone();
}

/**
 * @brief Aktualisiert die Zustände aller Kalman-Filter basierend auf den aktuellen Messungen.
 *
 * Für jeden Track wird der Kalman-Filter vorhergesagt und mit der aktuellen Position der Person
 * korrigiert. Die Position der Person wird basierend auf der Korrektur des Kalman-Filters aktualisiert.
 */
void MultiTracking::updateKalmanFilters() {
    for (auto& [trackId, person] : tracks) {
        cv::KalmanFilter& kalman = kalmanFilters[trackId];

        // Kalman-Vorhersage
        cv::Mat prediction = kalman.predict();

        // Dynamisches Rauschen basierend auf Geschwindigkeit
        float speed = std::sqrt(
            std::pow(kalman.statePost.at<float>(2), 2) + // dx^2
            std::pow(kalman.statePost.at<float>(3), 2)   // dy^2
        );
        float noiseScale = (speed > 10.0f) ? 1e-3 : 1e-4;
        setIdentity(kalman.processNoiseCov, cv::Scalar::all(noiseScale));

        // Messung (Position der Person)
        cv::Mat measurement(2, 1, CV_32F);
        cv::Point actualCentroid = person.getCentroid();
        measurement.at<float>(0) = actualCentroid.x;
        measurement.at<float>(1) = actualCentroid.y;

        // Kalman-Korrektur
        cv::Mat corrected = kalman.correct(measurement);

        // Korrigierte Position aktualisieren
        cv::Point correctedCentroid(
            static_cast<int>(corrected.at<float>(0)),
            static_cast<int>(corrected.at<float>(1))
        );
        person.setCentroid(correctedCentroid);
    }
}

/**
 * @brief Implementiert den ungarischen Algorithmus zur Zuordnung von Tracks zu Konturen.
 *
 * Implementiert den ungarischen Algorithmus, um die optimale Zuordnung zwischen bestehenden
 * Tracks und neuen Konturen basierend auf der Kostenmatrix zu finden. Gibt einen Vektor
 * zurück, der die Zuordnung von Tracks zu Konturen darstellt.
 *
 * @param costMatrix Die Kostenmatrix, die die Kosten für die Zuordnung jedes Tracks zu jeder Kontur enthält.
 * @return Ein Vektor von Zuordnungen, wobei der Index dem Track entspricht und der Wert die zugeordnete Kontur ist.
 */
std::vector<int> MultiTracking::hungarianAlgorithm(const std::vector<std::vector<double>>& costMatrix) {
    // Debugging: Kostenmatrix prüfen
    if (costMatrix.empty() || costMatrix[0].empty()) {
        std::cerr << "Fehler: Kostenmatrix ist leer." << std::endl;
        return {};
    }

    std::cout << "Kostenmatrix in HungarianAlgorithm:" << std::endl;
    for (size_t i = 0; i < costMatrix.size(); ++i) {
        for (size_t j = 0; j < costMatrix[i].size(); ++j) {
            std::cout << costMatrix[i][j] << " ";
            if (std::isnan(costMatrix[i][j]) || std::isinf(costMatrix[i][j])) {
                std::cerr << "Ungültiger Wert in Matrix: Zeile " << i 
                          << ", Spalte " << j << " = " << costMatrix[i][j] << std::endl;
            }
        }
        std::cout << std::endl;
    }

    int n = costMatrix.size();    // Anzahl der Tracks (Zeilen)
    int m = costMatrix[0].size(); // Anzahl der Konturen (Spalten)

    // Schritt 1: Zeilenreduktion
    std::vector<std::vector<double>> reducedMatrix = costMatrix;
    for (int i = 0; i < n; ++i) {
        double rowMin = *std::min_element(reducedMatrix[i].begin(), reducedMatrix[i].end());
        for (int j = 0; j < m; ++j) {
            reducedMatrix[i][j] -= rowMin;
        }
    }

    // Schritt 2: Spaltenreduktion
    for (int j = 0; j < m; ++j) {
        double colMin = std::numeric_limits<double>::max();
        for (int i = 0; i < n; ++i) {
            colMin = std::min(colMin, reducedMatrix[i][j]);
        }
        for (int i = 0; i < n; ++i) {
            reducedMatrix[i][j] -= colMin;
        }
    }

    // Zuordnung finden
    std::vector<int> assignment(n, -1);
    std::vector<bool> rowCovered(n, false);
    std::vector<bool> colCovered(m, false);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (reducedMatrix[i][j] == 0 && !rowCovered[i] && !colCovered[j]) {
                assignment[i] = j;
                rowCovered[i] = true;
                colCovered[j] = true;
                std::cout << "Zuordnung: Track " << i << " -> Contour " << j << std::endl;
            }
        }
    }

    return assignment;
}

/**
 * @brief Übergibt die verarbeiteten Daten an die Spiellogik.
 *
 * Vereinfachte Konturen werden berechnet und können an die Spiellogik weitergegeben werden.
 * Derzeit wird diese Funktion nur als Platzhalter verwendet und kann nach Bedarf erweitert werden.
 */
void MultiTracking::passDataToGameLogic() {
    for (const auto& track : tracks) {
        const Person& person = track.second;
        std::vector<cv::Point> simplifiedContour;
        cv::approxPolyDP(person.getContour(), simplifiedContour, 3, true);
        // Hier könnten weitere Schritte zur Übergabe der Daten an die Spiellogik folgen
    }
}

/**
 * @brief Visualisiert die Tracking-Ergebnisse auf dem gegebenen Frame.
 *
 * Zeichnet Bounding Boxes, IDs, Centroids und Keypoints für jede verfolgte Person auf das Frame.
 * Speichert das visualisierte Video und zeigt es in einem OpenCV-Fenster an.
 *
 * @param frame Das aktuelle Videoframe, das visualisiert werden soll.
 */
void MultiTracking::visualize(const cv::Mat& frame) const {
    static cv::VideoWriter videoWriter; // Statischer VideoWriter für die Speicherung
    static bool isWriterInitialized = false;

    // VideoWriter initialisieren
    if (!isWriterInitialized) {
        std::filesystem::create_directories("data/output"); // Erstelle den Ordner, falls er nicht existiert
        std::string outputPath = "data/output/tracking_output.avi";

        int frameWidth = frame.cols;
        int frameHeight = frame.rows;
        double fps = 30.0; // Standard-FPS
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // MJPG-Codec

        videoWriter.open(outputPath, codec, fps, cv::Size(frameWidth, frameHeight), true);
        if (!videoWriter.isOpened()) {
            std::cerr << "Fehler beim Initialisieren des VideoWriters!" << std::endl;
            return;
        }
        isWriterInitialized = true;
    }

    cv::Mat output = frame.clone(); // Klone das Frame, wenn Änderungen nötig sind

    // Prüfen, ob es Tracks gibt
    if (tracks.empty()) {
        // Leere Anzeige, falls keine Konturen/Tracks vorhanden sind
        cv::imshow("Tracking", output); // Zeige nur das ursprüngliche Frame
        return;
    }

    for (const auto& track : tracks) {
        const Person& person = track.second;

        // Hole den Begrenzungsrahmen (Bounding Box) der Kontur
        cv::Rect boundingBox = cv::boundingRect(person.getContour());

        // Farbauswahl basierend auf der ID
        cv::Scalar color = cv::Scalar(0, 255 - (person.getId() * 50) % 255, 255); // Unterschiedliche Farben für verschiedene IDs

        // Zeichne die Bounding Box
        cv::rectangle(output, boundingBox, color, 2); // Rechteck mit der definierten Farbe

        // Berechne die Position für die ID (leicht oberhalb der Bounding Box)
        cv::Point textPosition(boundingBox.x, boundingBox.y - 10); // 10 Pixel oberhalb der Bounding Box

        // Vermeide negative Textpositionen (falls Bounding Box am oberen Rand liegt)
        if (textPosition.y < 0) {
            textPosition.y = 10; // Setze die ID in den sichtbaren Bereich
        }

        // Zeige die ID der Person
        std::string text = "ID: " + std::to_string(person.getId());
        cv::putText(output, text, textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);

        // Zeichne den Schwerpunkt (Centroid) der Person
        cv::circle(output, person.getCentroid(), 5, cv::Scalar(255, 0, 0), -1);

        // Zeichne die Keypoints
        const auto& keypoints = person.getKeypoints(); // Liste der Keypoints der Person
        for (const auto& kp : keypoints) {
            cv::Point point(cvRound(kp.pt.x), cvRound(kp.pt.y)); // Umwandlung in cv::Point

            // Filter: Zeichne nur Keypoints innerhalb der Bounding-Box
            if (boundingBox.contains(point)) {
                cv::circle(output, point, 3, cv::Scalar(0, 255, 0), -1); // Grün für Keypoints
            }
        }
    }

    // Frame in das Video schreiben
    if (isWriterInitialized) {
        videoWriter.write(output);
    }

    // Zeige das aktualisierte Tracking-Ergebnis
    cv::imshow("Tracking", output);
}

/**
 * @brief Vergleicht zwei HSV-Histogramme und berechnet deren Ähnlichkeit.
 *
 * Verwendet die Korrelation (`HISTCMP_CORREL`), um die Ähnlichkeit zwischen zwei Histogrammen zu messen.
 *
 * @param hist1 Das erste HSV-Histogramm.
 * @param hist2 Das zweite HSV-Histogramm.
 * @return Der Ähnlichkeitswert zwischen den beiden Histogrammen.
 */
double MultiTracking::compareHistograms(const cv::Mat& hist1, const cv::Mat& hist2) {
    if (hist1.empty() || hist2.empty()) {
        std::cout << "Eines der Histogramme ist leer!" << std::endl;
        return 0.0;
    }

    // Ähnlichkeit berechnen
    double similarity = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);

    // Debugging-Ausgabe
    std::cout << "Histogram Vergleich - Ähnlichkeit: " << similarity << std::endl;

    return similarity;
}

/**
 * @brief Berechnet das HSV-Histogramm für eine gegebene Kontur innerhalb eines Frames.
 *
 * Extrahiert das ROI (Region of Interest) basierend auf der Bounding Box der Kontur,
 * konvertiert es in den HSV-Farbraum und berechnet das 2D-Histogramm für die
 * Hue- und Saturation-Kanäle. Normalisiert das Histogramm anschließend.
 *
 * @param frame Das aktuelle Videoframe.
 * @param contour Die Kontur, für die das Histogramm berechnet werden soll.
 * @return Das berechnete und normalisierte HSV-Histogramm.
 */
cv::Mat MultiTracking::calculateHsvHistogram(const cv::Mat& frame, const std::vector<cv::Point>& contour) {
    cv::Rect boundingBox = cv::boundingRect(contour);

    // Prüfen, ob die Bounding Box gültig ist
    if (boundingBox.area() == 0 || frame.empty()) {
        std::cout << "Ungültige Bounding Box oder leeres Frame." << std::endl;
        return cv::Mat();
    }

    // ROI extrahieren
    cv::Mat roi = frame(boundingBox);
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    // Histogrammberechnung
    int hBins = 50, sBins = 60;
    int histSize[] = {hBins, sBins};
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    const float* ranges[] = {hRanges, sRanges};
    int channels[] = {0, 1};
    cv::Mat hsvHistogram;

    cv::calcHist(&hsv, 1, channels, cv::Mat(), hsvHistogram, 2, histSize, ranges, true, false);
    cv::normalize(hsvHistogram, hsvHistogram, 0, 255, cv::NORM_MINMAX);

    // Debugging-Ausgabe: Zeige Histogramm
    std::cout << "Histogramm (erster Wert): " << (hsvHistogram.empty() ? 0 : hsvHistogram.at<float>(0, 0)) << std::endl;
    std::cout << "Histogramm Größe: " << hsvHistogram.rows << "x" << hsvHistogram.cols << std::endl;

    return hsvHistogram;
}

/**
 * @brief Berechnet und misst die Frames pro Sekunde (FPS) der Verarbeitung.
 *
 * Verwendet die Zeitstempel der Frames, um die aktuelle FPS zu berechnen und gibt diese
 * alle 1 Sekunde aus.
 *
 * @param frame Das aktuelle Videoframe (wird aktuell nicht verwendet, aber kann für zukünftige Erweiterungen genutzt werden).
 */
void MultiTracking::measureFPS(const cv::Mat& frame) {
    static auto lastTime = std::chrono::high_resolution_clock::now();
    static int frameCount = 0;
    static double fps = 0.0;

    frameCount++;

    // Berechne die Zeitdifferenz zum letzten Frame
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = currentTime - lastTime;

    if (elapsedTime.count() >= 1.0) { // Update alle 1 Sekunde
        fps = frameCount / elapsedTime.count();
        frameCount = 0;
        lastTime = currentTime;

        // Ausgabe der FPS
        std::cout << "FPS: " << fps << std::endl;
    }
}

/**
 * @brief Verarbeitet ein einzelnes Videoframe durch die gesamte Tracking-Pipeline.
 *
 * Führt alle Schritte der Tracking-Pipeline aus, einschließlich Hintergrundsubtraktion,
 * Konturenerkennung, Zuordnung von Konturen zu Tracks, Anwendung von Optical Flow,
 * Aktualisierung von Kalman-Filtern, Datenübergabe an die Spiellogik, Visualisierung
 * und FPS-Messung.
 *
 * @param frame Das aktuelle Videoframe, das verarbeitet werden soll.
 */
void MultiTracking::processFrame(const cv::Mat& frame) {
    static int frameCounter = 0; // Statischer Frame-Zähler

    static bool groundTruthLoaded = false;
    if (!groundTruthLoaded) {
       // loadGroundTruth("rsrc/annotations.xml", frameCounter);
        groundTruthLoaded = true;
    }

    // Frame-Zähler erhöhen
    frameCounter++;

    // Schritt 1: Hintergrundsubtraktion
    cv::Mat mask = applyKnn(frame);
    cv::imshow("applyKnn", mask); 

    // Bedingung: Nur wenn der Frame-Zähler größer als 30 ist, werden Schritt 3 und 4 ausgeführt
    if (frameCounter > 30) {
        // Schritt 3: Konturenerkennung
        std::vector<std::vector<cv::Point>> contours = findContours(mask);

        // Schritt 4: Konturen zu Tracks zuordnen
        assignContoursToTracks(contours, frame);
        // Schritt 5: Optical Flow anwenden
        applyOpticalFlow(frame, contours);
        updateKalmanFilters();
    }

    // Schritt 7: Daten übergeben
    passDataToGameLogic();

    // Schritt 8: Visualisierung
    visualize(frame);

    // FPS berechnen
    measureFPS(frame);

}

/**
 * @brief Lädt die Ground-Truth-Daten aus einer XML-Datei.
 *
 * Verwendet TinyXML2, um die XML-Datei zu laden und speichert die Bounding Boxes für jeden Frame
 * in der `groundTruthData`-Map.
 *
 * @param xmlFilePath Pfad zur XML-Datei mit den Ground-Truth-Daten.
 * @param frameCounter Referenz auf den Frame-Zähler (wird nicht verwendet und kann entfernt werden).
 */
void MultiTracking::loadGroundTruth(const std::string& xmlFilePath, int& frameCounter) {
    tinyxml2::XMLDocument doc;

    // Debug: Start des Ladevorgangs
    std::cout << "Starte das Laden der Ground-Truth-Daten aus: " << xmlFilePath << std::endl;

    // Laden der XML-Datei
    if (doc.LoadFile(xmlFilePath.c_str()) != tinyxml2::XML_SUCCESS) {
        std::cerr << "Fehler beim Laden der XML-Datei: " << xmlFilePath 
                  << ". Fehler: " << doc.ErrorName() << std::endl;
        return;
    }

    // Wurzelelement prüfen
    tinyxml2::XMLElement* root = doc.FirstChildElement("annotations");
    if (!root) {
        std::cerr << "Kein <annotations>-Element gefunden. Überprüfen Sie die XML-Struktur!" << std::endl;
        return;
    }

    // Debug: Anzahl der Tracks
    int trackCount = 0;

    // Iterieren durch alle Tracks in der XML-Datei
    for (tinyxml2::XMLElement* track = root->FirstChildElement("track"); track; track = track->NextSiblingElement("track")) {
        int id = track->IntAttribute("id");
        if (id < 0) {
            std::cerr << "Warnung: Track-ID ist negativ (" << id << "). Überspringe diesen Track." << std::endl;
            continue;
        }

        trackCount++;
        std::cout << "Lade Track mit ID: " << id << std::endl;

        int boxCount = 0; // Debug: Anzahl der Bounding Boxes pro Track

        // Iterieren durch alle Bounding Boxes des Tracks
        for (tinyxml2::XMLElement* box = track->FirstChildElement("box"); box; box = box->NextSiblingElement("box")) {
            int frame = box->IntAttribute("frame");
            float xtl = box->FloatAttribute("xtl");
            float ytl = box->FloatAttribute("ytl");
            float xbr = box->FloatAttribute("xbr");
            float ybr = box->FloatAttribute("ybr");

            // Debug: Bounding Box-Details
            std::cout << "  - Box: Frame=" << frame << ", xtl=" << xtl << ", ytl=" << ytl
                      << ", xbr=" << xbr << ", ybr=" << ybr << std::endl;

            // Validierung der Bounding Box-Werte
            if (xtl < 0 || ytl < 0 || xbr <= xtl || ybr <= ytl) {
                std::cerr << "  Warnung: Ungültige Bounding Box-Werte für Frame " << frame 
                          << ". Box wird übersprungen." << std::endl;
                continue;
            }

            // Bounding Box speichern
            BoundingBox bbox(frame, id, xtl, ytl, xbr, ybr);
            groundTruthData[frame].push_back(bbox);
            boxCount++;
        }

        // Debug: Anzahl der Bounding Boxes für den aktuellen Track
        std::cout << "  -> " << boxCount << " Bounding Boxes für Track ID " << id << " geladen." << std::endl;
    }

    // Debug: Gesamte Anzahl der Tracks und Bounding Boxes
    std::cout << "Ground-Truth-Ladevorgang abgeschlossen. Geladene Tracks: " << trackCount << std::endl;

    // Optional: Anzahl der Bounding Boxes pro Frame anzeigen
    for (const auto& [frame, boxes] : groundTruthData) {
        std::cout << "Frame " << frame << ": " << boxes.size() << " Bounding Boxes geladen." << std::endl;
    }
}

/**
 * @brief Berechnet die Intersection over Union (IoU) zwischen zwei Bounding Boxes.
 *
 * Die IoU misst die Überlappung zwischen zwei Bounding Boxes im Verhältnis zur Union ihrer Flächen.
 * Ein Wert von 1.0 bedeutet vollständige Überlappung, 0.0 keine Überlappung.
 *
 * @param box1 Erste Bounding Box.
 * @param box2 Zweite Bounding Box.
 * @return IoU-Wert zwischen den beiden Bounding Boxes.
 */
float MultiTracking::calculateIoU(const BoundingBox& box1, const BoundingBox& box2) const {
    // Berechnung der Schnittfläche
    float xIntersection = std::max(0.0f, std::min(box1.xbr, box2.xbr) - std::max(box1.xtl, box2.xtl));
    float yIntersection = std::max(0.0f, std::min(box1.ybr, box2.ybr) - std::max(box1.ytl, box2.ytl));
    float intersectionArea = xIntersection * yIntersection;

    // Berechnung der Vereinigungsfläche
    float unionArea = box1.area() + box2.area() - intersectionArea;

    // Vermeidung von Division durch Null
    if (unionArea == 0) {
        return 0.0f;
    }

    return intersectionArea / unionArea;
}

/**
 * @brief Testet die IoU-Berechnung zwischen getrackten Bounding Boxes und Ground-Truth-Daten.
 *
 * Durchläuft alle Frames und berechnet die IoU zwischen den Ground-Truth-Boxen und den vom Tracking-System
 * generierten Boxen. Gibt die Ergebnisse und die durchschnittliche IoU aus.
 */
void MultiTracking::testIoU() {
    std::cout << "Starte IoU-Test zwischen Tracking-Daten und Ground-Truth-Daten...\n";

    int frameCount = 0;
    float totalIoU = 0.0f;
    int matchedCount = 0;

    // Iteriere durch alle Frames in den Ground-Truth-Daten
    for (const auto& [frame, gtBoxes] : groundTruthData) {
        frameCount++;
        std::cout << "Frame " << frame << ":\n";

        // Sammle alle Bounding Boxes aus den Tracks
        std::vector<BoundingBox> trackedBoxes;
        for (const auto& [id, person] : tracks) {
            cv::Rect bbox = cv::boundingRect(person.getContour());
            trackedBoxes.emplace_back(frame, id, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height);
        }

        // Vergleiche jede Ground-Truth-Box mit jeder getrackten Box
        for (const auto& gtBox : gtBoxes) {
            float bestIoU = 0.0f;
            int bestMatchId = -1;

            for (const auto& trackedBox : trackedBoxes) {
                float iou = calculateIoU(gtBox, trackedBox);

                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestMatchId = trackedBox.id;
                }
            }

            std::cout << "  Ground-Truth Box (ID=" << gtBox.id << ") beste IoU mit Track ID "
                      << bestMatchId << ": " << bestIoU << "\n";

            if (bestIoU > 0.0f) {
                totalIoU += bestIoU;
                matchedCount++;
            }
        }
    }

    if (matchedCount > 0) {
        float avgIoU = totalIoU / matchedCount;
        std::cout << "Durchschnittliche IoU über alle Frames: " << avgIoU << "\n";
    } else {
        std::cout << "Keine übereinstimmenden Bounding Boxes gefunden.\n";
    }

    std::cout << "IoU-Test abgeschlossen.\n";
}


/**
 * @brief Berechnet die Multiple Object Tracking Accuracy (MOTA).
 * @return Der berechnete MOTA-Wert.
 *
 * MOTA berücksichtigt False Negatives (FN), False Positives (FP) und ID-Switches (IDSW) 
 * im Verhältnis zu den Ground-Truth-Objekten.

 * Ein höherer MOTA-Wert deutet auf eine bessere Tracking-Genauigkeit hin.
 */
float MultiTracking::calculateMOTA() {
    int totalFalseNegatives = 0;
    int totalFalsePositives = 0;
    int totalIDSwitches = 0;
    int totalGroundTruthObjects = 0;

    for (const auto& [frame, groundTruthBoxes] : groundTruthData) {
        totalGroundTruthObjects += groundTruthBoxes.size();

        std::vector<BoundingBox> trackedBoxes;
        for (const auto& [id, person] : tracks) {
            cv::Rect bbox = cv::boundingRect(person.getContour());
            trackedBoxes.emplace_back(frame, id, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height);
        }

        int falseNegatives = 0;
        int falsePositives = 0;
        int idSwitches = 0;

        std::map<int, int> groundTruthToTrack;
        for (const auto& gtBox : groundTruthBoxes) {
            float bestIoU = 0.0f;
            int bestTrackId = -1;

            for (const auto& trackedBox : trackedBoxes) {
                float iou = calculateIoU(gtBox, trackedBox);
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestTrackId = trackedBox.id;
                }
            }

            if (bestIoU >= 0.5) {
                if (groundTruthToTrack.count(gtBox.id) > 0 && groundTruthToTrack[gtBox.id] != bestTrackId) {
                    idSwitches++;
                }
                groundTruthToTrack[gtBox.id] = bestTrackId;
            } else {
                falseNegatives++;
            }
        }

        for (const auto& trackedBox : trackedBoxes) {
            bool matched = false;
            for (const auto& [gtId, trackId] : groundTruthToTrack) {
                if (trackId == trackedBox.id) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                falsePositives++;
            }
        }

        totalFalseNegatives += falseNegatives;
        totalFalsePositives += falsePositives;
        totalIDSwitches += idSwitches;
    }

    if (totalGroundTruthObjects == 0) {
        return 0.0f;
    }

    return 1.0f - static_cast<float>(totalFalseNegatives + totalFalsePositives + totalIDSwitches) / totalGroundTruthObjects;
}


/**
 * @brief Berechnet den F1-Score für das Tracking.
 * @return Der berechnete F1-Score.
 *
 * Der F1-Score ist das harmonische Mittel von Präzision und Recall und gibt eine ausgewogene
 * Metrik für die Tracking-Performance.
 *
 */
float MultiTracking::calculateF1Score() {
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;

    for (const auto& [frame, groundTruthBoxes] : groundTruthData) {
        std::vector<BoundingBox> trackedBoxes;
        for (const auto& [id, person] : tracks) {
            cv::Rect bbox = cv::boundingRect(person.getContour());
            trackedBoxes.emplace_back(frame, id, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height);
        }

        std::set<int> matchedGroundTruth;
        std::set<int> matchedTracked;

        for (size_t i = 0; i < groundTruthBoxes.size(); ++i) {
            const BoundingBox& gtBox = groundTruthBoxes[i];
            float bestIoU = 0.0f;
            int bestMatchIdx = -1;

            for (size_t j = 0; j < trackedBoxes.size(); ++j) {
                const BoundingBox& trackedBox = trackedBoxes[j];
                float iou = calculateIoU(gtBox, trackedBox);

                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestMatchIdx = j;
                }
            }

            if (bestIoU >= 0.5) {
                matchedGroundTruth.insert(i);
                matchedTracked.insert(bestMatchIdx);
                truePositives++;
            }
        }

        falsePositives += trackedBoxes.size() - matchedTracked.size();
        falseNegatives += groundTruthBoxes.size() - matchedGroundTruth.size();
    }

    float precision = truePositives > 0 ? static_cast<float>(truePositives) / (truePositives + falsePositives) : 0.0f;
    float recall = truePositives > 0 ? static_cast<float>(truePositives) / (truePositives + falseNegatives) : 0.0f;

    if (precision + recall == 0.0f) {
        return 0.0f;
    }

    return 2.0f * (precision * recall) / (precision + recall);
}


/**
 * @brief Berechnet die Mean Average Precision (mAP) für das Tracking.
 * @return Der berechnete mAP-Wert.
 *
 * Die mAP ist ein Durchschnitt der Average Precision (AP) über alle Klassen oder Objekte.
 * Sie berücksichtigt die Präzision und den Recall bei verschiedenen IoU-Schwellen.
 */
float MultiTracking::calculateMeanAveragePrecision() {
    int totalGroundTruths = 0;
    float sumAveragePrecision = 0.0f;

    // Iteriere über alle Frames
    for (const auto& [frame, groundTruthBoxes] : groundTruthData) {
        // Hole die getrackten Bounding Boxes für das aktuelle Frame
        std::vector<BoundingBox> trackedBoxes;
        for (const auto& [id, person] : tracks) {
            cv::Rect bbox = cv::boundingRect(person.getContour());
            trackedBoxes.emplace_back(frame, id, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height);
        }

        // Berechnung von Precision und Recall für die Tracking-Boxen
        std::vector<float> precisions;
        std::vector<float> recalls;

        int truePositives = 0;
        int falsePositives = 0;

        std::set<int> matchedGroundTruth;

        for (const auto& trackedBox : trackedBoxes) {
            float bestIoU = 0.0f;
            int bestMatchIdx = -1;

            for (size_t i = 0; i < groundTruthBoxes.size(); ++i) {
                if (matchedGroundTruth.find(i) != matchedGroundTruth.end()) {
                    continue; // Überspringe bereits gematchte Ground-Truth-Boxen
                }

                float iou = calculateIoU(trackedBox, groundTruthBoxes[i]);
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestMatchIdx = i;
                }
            }

            if (bestIoU >= 0.5) { // IoU-Schwelle von 0.5
                truePositives++;
                matchedGroundTruth.insert(bestMatchIdx);
            } else {
                falsePositives++;
            }

            int falseNegatives = groundTruthBoxes.size() - matchedGroundTruth.size();
            float precision = truePositives / static_cast<float>(truePositives + falsePositives);
            float recall = truePositives / static_cast<float>(truePositives + falseNegatives);

            precisions.push_back(precision);
            recalls.push_back(recall);
        }

        // Berechnung der Average Precision (AP) für das aktuelle Frame
        float averagePrecision = 0.0f;
        for (size_t i = 1; i < precisions.size(); ++i) {
            averagePrecision += precisions[i] * (recalls[i] - recalls[i - 1]);
        }

        sumAveragePrecision += averagePrecision;
        totalGroundTruths += groundTruthBoxes.size();
    }

    // Mittelwert der Average Precision berechnen
    if (totalGroundTruths == 0) {
        return 0.0f; // Keine Ground-Truth-Daten
    }

    return sumAveragePrecision / totalGroundTruths;
}
