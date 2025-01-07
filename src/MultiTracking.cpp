#include "MultiTracking.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

// Konstruktor
MultiTracking::MultiTracking() : nextId(0), trackingPipeline() {}


// Destruktor
MultiTracking::~MultiTracking() {}


void MultiTracking::initializeKalmanFilter(Person& person) {
    cv::KalmanFilter kalman(4, 2, 0); // 4 Zustandsvariablen (x, y, dx, dy), 2 Messvariablen (x, y)

    // Initialisiere den Zustand (x, y, dx, dy)
    cv::Point centroid = person.getCentroid();
    kalman.statePre.at<float>(0) = centroid.x;
    kalman.statePre.at<float>(1) = centroid.y;
    kalman.statePre.at<float>(2) = 0; // Geschwindigkeit in x
    kalman.statePre.at<float>(3) = 0; // Geschwindigkeit in y

    // Messmatrix (wir messen nur Position)
    kalman.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);

    // Übergangsmatrix (lineares Modell)
    kalman.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,  // x = x + dx
        0, 1, 0, 1,  // y = y + dy
        0, 0, 1, 0,  // dx bleibt gleich
        0, 0, 0, 1); // dy bleibt gleich

    // Prozessrauschen
    setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-4));
    setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-2)); // Messrauschen
    setIdentity(kalman.errorCovPost, cv::Scalar::all(1));           // Fehler im Nachzustand

    kalmanFilters[person.getId()] = kalman; // Speichere den Kalman-Filter
}


void MultiTracking::processFrame(const cv::Mat& frame) {
    // Schritt 1: Hintergrundsubtraktion
    cv::Mat mask = applyKnn(frame);

    // Schritt 2: Überlappungshandhabung
    cv::Mat segmented = performWatershed(frame, mask);

    // Schritt 3: Konturenerkennung
    std::vector<std::vector<cv::Point>> contours = findContours(segmented);

    // Schritt 4: Optical Flow anwenden
    applyOpticalFlow(frame);

    // Schritt 5: Kalman-Filter aktualisieren
    updateKalmanFilters();

    // Schritt 6: Konturen zu Tracks zuordnen
    assignContoursToTracks(contours, frame);

    // Schritt 7: Daten übergeben
    passDataToGameLogic();

    // Schritt 8: Visualisierung
    visualize(frame);

    // FPS berechnen
    measureFPS(frame);

}
// Schritt 1: Hintergrundsubtraktion
cv::Mat MultiTracking::applyKnn(const cv::Mat& frame) {
    return trackingPipeline.applyKnn(frame);
}


// Schritt 2: Überlappungshandhabung
cv::Mat MultiTracking::performWatershed(const cv::Mat& frame, const cv::Mat& mask) {
    cv::Mat distTransform;
    cv::distanceTransform(mask, distTransform, cv::DIST_L2, 5);
    cv::normalize(distTransform, distTransform, 0, 1.0, cv::NORM_MINMAX);

    // Marker erstellen
    cv::Mat markers;
    cv::threshold(distTransform, markers, 0.5, 1.0, cv::THRESH_BINARY); // Binärmaske erstellen
    markers.convertTo(markers, CV_8U); // In 8-Bit konvertieren, falls nötig

    // Verbundene Komponenten identifizieren
    cv::Mat connectedComponents;
    cv::connectedComponents(markers, connectedComponents, 8, CV_32S);

    // Konvertiere für den Wasserscheiden-Algorithmus
    connectedComponents += 1; // Hintergrund auf 1 setzen
    cv::watershed(frame, connectedComponents);

    // Erstelle segmentierte Maske
    cv::Mat segmented;
    connectedComponents.convertTo(segmented, CV_8U); // Optional für Visualisierung
    return segmented;
}



// Schritt 3: Konturenerkennung
std::vector<std::vector<cv::Point>> MultiTracking::findContours(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> allContours;
    std::vector<std::vector<cv::Point>> filteredContours;
    
    // Finde alle Konturen
    cv::findContours(mask, allContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Filtere kleine Konturen basierend auf ihrer Fläche
    const double minArea = 5000.0; // Mindestfläche für relevante Konturen
    for (const auto& contour : allContours) {
        if (cv::contourArea(contour) >= minArea) {
            filteredContours.push_back(contour);
        }
    }

    return filteredContours; // Rückgabe nur der gefilterten Konturen
}

// Schritt 4: Optical Flow anwenden
void MultiTracking::applyOpticalFlow(const cv::Mat& frame) {
    if (prevGrayFrame.empty()) {
        cv::cvtColor(frame, prevGrayFrame, cv::COLOR_BGR2GRAY);
        return;
    }

    cv::Mat currGrayFrame;
    cv::cvtColor(frame, currGrayFrame, cv::COLOR_BGR2GRAY);

    for (auto& [trackId, person] : tracks) {
        std::vector<cv::Point2f> trackedPoints = person.getTrackedPoints();
        if (trackedPoints.empty()) continue;

        std::vector<cv::Point2f> nextPoints;
        std::vector<uchar> status;
        std::vector<float> err;

        // Optical Flow für Keypoints berechnen
        cv::calcOpticalFlowPyrLK(prevGrayFrame, currGrayFrame, trackedPoints, nextPoints, status, err);

        // Entferne ungültige Keypoints
        std::vector<cv::Point2f> validNextPoints;
        for (size_t i = 0; i < nextPoints.size(); ++i) {
            if (status[i]) {
                validNextPoints.push_back(nextPoints[i]);
            }
        }

        // Aktualisiere die getrackten Keypoints
        person.setTrackedPoints(validNextPoints);

        // Optional: Schwerpunkt aktualisieren
        if (!validNextPoints.empty()) {
            cv::Point2f newCentroid(0, 0);
            for (const auto& pt : validNextPoints) {
                newCentroid += pt;
            }
            newCentroid.x /= validNextPoints.size();
            newCentroid.y /= validNextPoints.size();
            person.setCentroid(cv::Point(static_cast<int>(newCentroid.x), static_cast<int>(newCentroid.y)));
        }
    }

    prevGrayFrame = currGrayFrame.clone();
}


// Schritt 5: Kalman-Filter aktualisieren
void MultiTracking::updateKalmanFilters() {
    for (auto& [trackId, person] : tracks) {
        // Kalman-Filter Vorhersage
        cv::KalmanFilter& kalman = kalmanFilters[trackId];
        cv::Mat prediction = kalman.predict();

        // Vorhergesagte Position
        cv::Point predictedCentroid(
            static_cast<int>(prediction.at<float>(0)),
            static_cast<int>(prediction.at<float>(1))
        );

        // Messung aktualisieren
        cv::Mat measurement(2, 1, CV_32F);
        cv::Point actualCentroid = person.getCentroid();
        measurement.at<float>(0) = actualCentroid.x;
        measurement.at<float>(1) = actualCentroid.y;

        // Kalman-Filter korrigieren
        cv::Mat corrected = kalman.correct(measurement);

        // Korrigierte Position
        cv::Point correctedCentroid(
            static_cast<int>(corrected.at<float>(0)),
            static_cast<int>(corrected.at<float>(1))
        );

        // Aktualisiere die Position im Person-Objekt
        person.setCentroid(correctedCentroid);
    }
}


// Schritt 6: Konturen zu Tracks zuordnen
void MultiTracking::assignContoursToTracks(const std::vector<std::vector<cv::Point>>& contours, const cv::Mat& frame) {
    std::vector<std::vector<double>> costMatrix(tracks.size(), std::vector<double>(contours.size(), 0));

    int i = 0;
    for (const auto& [trackId, person] : tracks) {
        int j = 0;
        for (const auto& contour : contours) {
            cv::Point trackCentroid = person.getCentroid();
            cv::Moments moments = cv::moments(contour);
            cv::Point contourCentroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

            // Berechne die euklidische Distanz
            double distance = cv::norm(trackCentroid - contourCentroid);

            // Vergleich der Fläche
            double contourArea = cv::contourArea(contour);
            double areaDifference = std::abs(person.getArea() - contourArea);

            // Vergleich des Aspektverhältnisses
            cv::Rect boundingBox = cv::boundingRect(contour);
            double contourAspectRatio = static_cast<double>(boundingBox.width) / boundingBox.height;
            double aspectRatioDifference = std::abs(person.getAspectRatio() - contourAspectRatio);

            // Vergleich des HSV-Histogramms
            cv::Mat contourHistogram = MultiTracking::calculateHsvHistogram(frame, contour);

            double histogramSimilarity = compareHistograms(person.getHsvHistogram(), contourHistogram);

            // Kombinierte Kosten berechnen
            double totalCost = 0.5 * distance + 0.3 * areaDifference + 0.1 * aspectRatioDifference + 0.1 * (1 - histogramSimilarity);
            costMatrix[i][j] = totalCost;

            j++;
        }
        i++;
    }

    // Zuordnung mit Ungarischer Methode
    std::vector<int> assignment(tracks.size(), -1);
    for (size_t t = 0; t < tracks.size(); ++t) {
        double minCost = 1e9;
        int bestContour = -1;
        for (size_t c = 0; c < contours.size(); ++c) {
            if (costMatrix[t][c] < minCost) {
                minCost = costMatrix[t][c];
                bestContour = c;
            }
        }
        assignment[t] = bestContour;
    }

    for (size_t t = 0; t < assignment.size(); ++t) {
        if (assignment[t] != -1) {
            int contourIdx = assignment[t];
            auto& person = tracks[t];
            person.setContour(contours[contourIdx]);
            person.extractKeypoints(contours[contourIdx]); // Keypoints extrahieren
        }
    }

    // Neue Konturen, die keinem Track zugeordnet wurden
    for (size_t c = 0; c < contours.size(); ++c) {
        if (std::find(assignment.begin(), assignment.end(), c) == assignment.end()) {
            int newId = nextId++;
            Person newPerson(newId, contours[c]);
            initializeKalmanFilter(newPerson);
            tracks[newId] = newPerson;
        }
    }
}

// Schritt 7: Daten übergeben
void MultiTracking::passDataToGameLogic() {
    for (const auto& track : tracks) {
        const Person& person = track.second;
        std::vector<cv::Point> simplifiedContour;
        cv::approxPolyDP(person.getContour(), simplifiedContour, 3, true);
    }
}


// Schritt 8: Visualisierung
void MultiTracking::visualize(const cv::Mat& frame) const {
    cv::Mat output = frame.clone(); // Klone das Frame, wenn Änderungen nötig sind
    for (const auto& track : tracks) {
        const Person& person = track.second;

        cv::drawContours(output, std::vector<std::vector<cv::Point>>{person.getContour()}, -1, cv::Scalar(0, 255, 0), 2);
        cv::circle(output, person.getCentroid(), 5, cv::Scalar(255, 0, 0), -1);
        cv::putText(output, std::to_string(person.getId()), person.getCentroid(),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
    }
    cv::imshow("Tracking", output); // Debug-Ausgabe
}


double MultiTracking::compareHistograms(const cv::Mat& hist1, const cv::Mat& hist2) {
    if (hist1.empty() || hist2.empty()) {
        return 0.0; // Geringe Ähnlichkeit, falls ein Histogramm fehlt
    }
    return cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
}


cv::Mat MultiTracking::calculateHsvHistogram(const cv::Mat& frame, const std::vector<cv::Point>& contour) {
    cv::Rect boundingBox = cv::boundingRect(contour);
    if (boundingBox.area() == 0 || frame.empty()) {
        return cv::Mat();
    }
    cv::Mat roi = frame(boundingBox);
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    int hBins = 50, sBins = 60;
    int histSize[] = {hBins, sBins};
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    const float* ranges[] = {hRanges, sRanges};
    int channels[] = {0, 1};
    cv::Mat hsvHistogram;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hsvHistogram, 2, histSize, ranges, true, false);
    cv::normalize(hsvHistogram, hsvHistogram, 0, 255, cv::NORM_MINMAX);
    return hsvHistogram;
}

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