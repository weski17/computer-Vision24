#include <SFML/Graphics.hpp>
#include "TrackingPipeline.hpp"
#include <opencv2/opencv.hpp>

int main() {
    const int windowWidth = 800;
    const int windowHeight = 600;

    // Erstelle ein Fenster für die Benutzeroberfläche
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Live Tracking");

    // Initialisiere Tracking-Pipeline
    TrackingPipeline tracking;

    // Öffne Webcam (Index 0 für Standardkamera)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Fehler: Webcam konnte nicht geöffnet werden!" << std::endl;
        return -1;
    }

    // Hauptverarbeitungsschleife
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        // Lade aktuellen Frame von der Webcam
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Fehler: Kein Frame von der Webcam erhalten!" << std::endl;
            break;
        }

        // Verarbeite Frame mit der Tracking-Pipeline
        tracking.processFrame(frame);

        // Abrufen des Konturs der verfolgten Person
        const std::vector<cv::Point>& contour = tracking.getTrackedContour();
        if (!contour.empty()) {
            // Kontur auf dem Frame zeichnen
            std::vector<std::vector<cv::Point>> contoursToDraw = {contour};
            cv::drawContours(frame, contoursToDraw, -1, cv::Scalar(0, 255, 0), 2); // Grün
        }

        // Zeige das Ergebnis im OpenCV-Fenster
        cv::imshow("Live Tracking Output", frame);

        // Beende bei ESC-Taste
        if (cv::waitKey(1) == 27) {
            break;
        }

        // Aktualisiere das SFML-Fenster (optional, falls UI benötigt wird)
        window.clear();
        window.display();
    }

    // Ressourcen freigeben
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
