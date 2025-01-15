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


// Schritt 1: Hintergrundsubtraktion
cv::Mat MultiTracking::applyKnn(const cv::Mat& frame) {
    return trackingPipeline.applyKnn(frame);
}

// Schritt 3: Konturenerkennung
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

        if (trackedPoints.empty() || trackedPoints.size() < MIN_TRACKED_POINTS) {
            // Wenn zu wenige Punkte vorhanden sind, extrahiere neue Keypoints
            const auto& contour = person.getContour(); // Hole die Kontur der Person
            person.extractKeypoints(contour);

            // Konvertiere Keypoints in trackedPoints
            std::vector<cv::Point2f> newTrackedPoints;
            for (const auto& keypoint : person.getKeypoints()) {
                newTrackedPoints.push_back(keypoint.pt);
            }

            person.setTrackedPoints(newTrackedPoints);
            trackedPoints = newTrackedPoints; // Aktualisiere für diesen Frame
        }

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

    std::cout << "Dimensionen der Matrix: " << n << " x " << m << std::endl;

    // Schritt 1: Zeilenreduktion
    std::vector<std::vector<double>> reducedMatrix = costMatrix;
    for (int i = 0; i < n; ++i) {
        double rowMin = *std::min_element(reducedMatrix[i].begin(), reducedMatrix[i].end());
        for (int j = 0; j < m; ++j) {
            reducedMatrix[i][j] -= rowMin;
        }
    }

    std::cout << "Matrix nach Zeilenreduktion:" << std::endl;
    for (const auto& row : reducedMatrix) {
        for (double cost : row) {
            std::cout << cost << " ";
        }
        std::cout << std::endl;
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

    std::cout << "Matrix nach Spaltenreduktion:" << std::endl;
    for (const auto& row : reducedMatrix) {
        for (double cost : row) {
            std::cout << cost << " ";
        }
        std::cout << std::endl;
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




// Schritt 6: Konturen zu Tracks zuordnen
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
            double totalCost = 0.7 * distance + 0.2 * areaDifference + 0.1 * aspectRatioDifference + 0.3 * (1 - histogramSimilarity);   
            if (totalCost > 500.0) {
            totalCost = 1e9; // Setzen Sie sehr hohe Kosten für unplausible Zuordnungen
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
            person.setContour(contours[contourIdx]);
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

            // Debugging: Neue Tracks hinzufügen
            std::cout << "Neuer Track hinzugefügt: ID " << newId << " für Kontur " << c << std::endl;

            usedContours.push_back(c); // Markiere diese Kontur als verwendet
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

        // Hole den Begrenzungsrahmen (Bounding Box) der Kontur
        cv::Rect boundingBox = cv::boundingRect(person.getContour());

        // Farbauswahl basierend auf der ID
        cv::Scalar color = cv::Scalar(0, 255 - person.getId() * 50, 255); // Unterschiedliche Farben für verschiedene IDs

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
    }

    cv::imshow("Tracking", output); // Zeige das Tracking-Ergebnis
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

void MultiTracking::processFrame(const cv::Mat& frame) {
    // Schritt 1: Hintergrundsubtraktion
    cv::Mat mask = applyKnn(frame);
    cv::imshow("applyKnn", mask); 

    // Schritt 3: Konturenerkennung
    std::vector<std::vector<cv::Point>> contours = findContours(mask);

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